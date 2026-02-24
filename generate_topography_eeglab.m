function generate_topography_eeglab(dataFile, outputDir)
% GENERATE_TOPOGRAPHY_EEGLAB - EEG topography generator using EEGLAB.

%% 1. Configuration Priority (Args > JSON > Defaults)
configPath = 'config.json';
cfg = default_config();


if exist(configPath, 'file')
    try
        fileCfg = jsondecode(fileread(configPath));
        if isfield(fileCfg, 'excel_file'), cfg.excel_file = fileCfg.excel_file; end
        if isfield(fileCfg, 'output_dir_matlab'), cfg.output_dir_matlab = fileCfg.output_dir_matlab; end
        if isfield(fileCfg, 'electrodes'), cfg.electrodes = fileCfg.electrodes; end
        if isfield(fileCfg, 'bands'), cfg.bands = fileCfg.bands; end
        if isfield(fileCfg, 'eeglab_path'), cfg.eeglab_path = fileCfg.eeglab_path; end
        if isfield(fileCfg, 'regions'), cfg.regions = fileCfg.regions; end
        if isfield(fileCfg, 'visualization')
            vis = fileCfg.visualization;
            flds = fieldnames(vis);
            for fi = 1:length(flds)
                cfg.visualization.(flds{fi}) = vis.(flds{fi});
            end
        end
    catch
        warning('Could not parse config.json. Using defaults.');
    end
end

if nargin < 1 || isempty(dataFile), dataFile = detect_data_file(cfg.excel_file); end
if nargin < 2 || isempty(outputDir), outputDir = cfg.output_dir_matlab; end
electrodes = cfg.electrodes;
if iscell(electrodes) && size(electrodes, 1) > 1, electrodes = electrodes'; end
bands = cfg.bands;
if iscell(bands) && size(bands, 1) > 1, bands = bands'; end

fprintf('Starting EEGLAB Topography Pipeline...\n');

%% 2. EEGLAB Detection & Path Injection
originalPath = path;
cleanupObj = onCleanup(@() path(originalPath));


if isempty(which('topoplot'))
    fprintf('EEGLAB not initialized. Searching for eeglab.m...\n');


    eeglabFile = which('eeglab.m');
    if isempty(eeglabFile)
        searchPaths = {cfg.eeglab_path, 'C:\MATLAB\eeglab', ...
            fullfile(char(java.lang.System.getProperty('user.home')), 'Documents', 'MATLAB', 'eeglab'), ...
            pwd};
        for sIdx = 1:length(searchPaths)
            if ~isempty(searchPaths{sIdx}) && exist(fullfile(searchPaths{sIdx}, 'eeglab.m'), 'file')
                eeglabFile = fullfile(searchPaths{sIdx}, 'eeglab.m');
                addpath(searchPaths{sIdx});
                break;
            end
        end
    end

    if isempty(eeglabFile)
        error('EEGLAB not found. Set "eeglab_path" in config.json.');
    end

    fprintf('  Found EEGLAB at: %s\n', eeglabFile);
    eeglab nogui;
else
    fprintf('  EEGLAB is already initialized.\n');
    eeglabFile = which('eeglab.m');
end

% Inject critical EEGLAB resource folders
eeglabDir = fileparts(eeglabFile);
criticalSubdirs = {'functions/resources', 'functions/sigprocfunc', 'functions/popfunc', ...
    'sample_locs', 'functions/adminfunc', 'functions/supportfiles/channel_location_files/eeglab'};
for i = 1:length(criticalSubdirs)
    fullSub = fullfile(eeglabDir, strrep(criticalSubdirs{i}, '/', filesep));
    if exist(fullSub, 'dir')
        addpath(fullSub);
    end
end

%% 3. Validation
if ~exist(dataFile, 'file')
    error('Input error: File not found: "%s"', dataFile);
end

%% 4. Setup Channel Locations
fprintf('Setting up channel locations...\n');
chanlocs = setup_chanlocs_robust(electrodes, eeglabDir);

if isempty(chanlocs) || ~isfield(chanlocs, 'theta') || ~isfield(chanlocs, 'radius')
    error('FATAL: Channel coordinates could not be retrieved. Coordinates are required for topoplot.');
end
fprintf('  Validation passed: %d electrodes have theta/radius coordinates.\n', length(chanlocs));

%% 5. Data Processing
fprintf('Loading and averaging data from %s...\n', dataFile);
[~,~,ext] = fileparts(dataFile);
switch lower(ext)
    case '.xlsx'
        dataTable = readtable(dataFile);
    case '.csv'
        dataTable = readtable(dataFile, 'Delimiter', ',');
    case '.txt'
        dataTable = readtable(dataFile, 'Delimiter', '\t');
    otherwise
        error('Unsupported file format: %s', ext);
end
[bandAverages, stats] = process_band_data(dataTable, bands, electrodes);
print_summary(stats);

%% 6. Visualization
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
fmt = get_vis(cfg, 'output_format', 'png');
fprintf('Exporting plots to "%s/" (format: %s)...\n', outputDir, fmt);

plots = get_vis(cfg, 'plots', struct());


if get_flag(plots, 'topoplot', true) || get_flag(plots, 'topoplot_combined', true)
    generate_topoplots(bandAverages, chanlocs, outputDir, cfg);
end


if get_flag(plots, 'regional_bar', true)
    generate_regional_bar_chart(bandAverages, electrodes, cfg, outputDir);
end


if get_flag(plots, 'heatmap', true)
    generate_heatmap(bandAverages, electrodes, cfg, outputDir);
end

fprintf('\nPipeline completed successfully.\n');
end



function chanlocs = setup_chanlocs_robust(electrodeLabels, eeglabDir)
% SETUP_CHANLOCS_ROBUST  Map electrode labels to template coordinates.
%   Handles T3/T4/T5/T6 → T7/T8/P7/P8 aliasing for the Standard-10-20-Cap19.ced
%   template, which uses modern nomenclature. Preserves ALL struct fields
%   (theta, radius, sph_theta, sph_phi, sph_radius, X, Y, Z) so that
%   topoplot can perform its 2D polar projection correctly.

chanlocs = [];

templates = {'Standard-10-20-Cap19.ced', 'standard-10-20-cap19.elp', ...
    'standard-10-5-cap385.elp', 'standard_BESA.elp'};

% T3/T4/T5/T6 → T7/T8/P7/P8 aliasing
aliases = containers.Map( ...
    {'T3','T4','T5','T6'}, ...
    {'T7','T8','P7','P8'});

templatePath = '';


for k = 1:length(templates)
    templatePath = which(templates{k});
    if ~isempty(templatePath), break; end
end


if isempty(templatePath)
    fprintf('  Searching EEGLAB directory for coordinate templates...\n');
    for k = 1:length(templates)
        results = dir(fullfile(eeglabDir, '**', templates{k}));
        if ~isempty(results)
            templatePath = fullfile(results(1).folder, results(1).name);
            break;
        end
    end
end

if isempty(templatePath)
    warning('topography:no_template', 'No coordinate template file found.');
    return;
end


fprintf('  Using template: %s\n', templatePath);
try
    allLocs = readlocs(templatePath);
catch ME
    warning('topography:readlocs_fail', 'readlocs failed: %s', ME.message);
    return;
end

templateLabels = upper({allLocs.labels});
nElectrodes = length(electrodeLabels);
matchIndices = zeros(1, nElectrodes);
originalLabels = electrodeLabels;

for k = 1:nElectrodes
    label = electrodeLabels{k};
    lookupLabel = label;

    if aliases.isKey(label)
        lookupLabel = aliases(label);
    end

    idx = find(strcmpi(templateLabels, lookupLabel), 1);
    if ~isempty(idx)
        matchIndices(k) = idx;
    else
        fprintf('  WARNING: Electrode "%s" (lookup: "%s") not found in template.\n', label, lookupLabel);
    end
end

foundMask = matchIndices > 0;
nFound = sum(foundMask);

if nFound == 0
    warning('topography:no_match', 'No electrodes matched the template.');
    return;
end

% Index directly into allLocs to preserve all fields (theta, radius, X, Y, Z, etc.)
chanlocs = allLocs(matchIndices(foundMask));

% Restore original clinical labels
foundLabels = originalLabels(foundMask);
for k = 1:length(chanlocs)
    chanlocs(k).labels = foundLabels{k};
end

fprintf('  Mapped %d/%d electrodes. Labels: %s\n', nFound, nElectrodes, strjoin({chanlocs.labels}, ', '));


fprintf('  --- Coordinate verification ---\n');
for k = 1:length(chanlocs)
    fprintf('    %-4s  theta=%7.1f  radius=%.3f\n', chanlocs(k).labels, chanlocs(k).theta, chanlocs(k).radius);
end

if nFound < nElectrodes
    missing = originalLabels(~foundMask);
    fprintf('  WARNING: Missing electrodes will be excluded: %s\n', strjoin(missing, ', '));
    setappdata(0, 'chanlocs_foundMask', foundMask);
else
    setappdata(0, 'chanlocs_foundMask', true(1, nElectrodes));
end
end

function [bandAverages, summaries] = process_band_data(tbl, bands, electrodes)
bandAverages = struct();
summaries = struct();
for bIdx = 1:length(bands)
    band = bands{bIdx};
    values = zeros(1, length(electrodes));
    for eIdx = 1:length(electrodes)
        colName = sprintf('%s_%s', band, electrodes{eIdx});
        if ismember(colName, tbl.Properties.VariableNames)
            values(eIdx) = mean(tbl.(colName), 'omitnan');
        else
            values(eIdx) = NaN;
            warning('topography:missing_col', 'Column "%s" missing.', colName);
        end
    end
    bandAverages.(band) = values;
    summaries.(band).min = min(values);
    summaries.(band).max = max(values);
    summaries.(band).mean = mean(values, 'omitnan');
end
end

function generate_topoplots(averages, chanlocs, outputDir, cfg)
bands = fieldnames(averages);
fmt = get_vis(cfg, 'output_format', 'png');
plots = get_vis(cfg, 'plots', struct());
cmap = resolve_colormap(cfg);
titleSize = get_vis(cfg, 'font_size_title', 16);
nContours = get_vis(cfg, 'num_contours', 6);
estyle = get_vis(cfg, 'electrode_style', 'labels');
powerLabel = get_vis(cfg, 'power_label', 'Absolute Power (µV²)');


if get_flag(plots, 'topoplot_combined', true)
    hCombined = figure('Color', 'w', 'Position', [100, 100, 1100, 1100], 'Visible', 'off');
    sgtitle('Grand Average EEG Topography', 'FontSize', 18, 'FontWeight', 'bold');
    numB = length(bands);
    nR = ceil(sqrt(numB)); nC = ceil(numB/nR);
    for k = 1:numB
        subplot(nR, nC, k);
        render_topography(averages.(bands{k}), chanlocs, bands{k}, 14, cmap, nContours, estyle, powerLabel);
    end
    save_figure(hCombined, outputDir, 'combined_topography_eeglab', fmt);
end


if get_flag(plots, 'topoplot', true)
    for k = 1:length(bands)
        hSingle = figure('Color', 'w', 'Position', [100, 100, 600, 600], 'Visible', 'off');
        render_topography(averages.(bands{k}), chanlocs, bands{k}, titleSize, cmap, nContours, estyle, powerLabel);
        save_figure(hSingle, outputDir, sprintf('%s_eeglab_topomap', bands{k}), fmt);
    end
end
end

function render_topography(data, chanlocs, bandName, titleSize, cmap, nContours, estyle, powerLabel)
topoplot(data, chanlocs, 'style', 'both', 'electrodes', estyle, ...
    'numcontour', nContours, ...
    'maplimits', [min(data), max(data)], 'colormap', cmap);
formatted = [upper(bandName(1)), lower(bandName(2:end))];
title(sprintf('%s Band', formatted), 'FontSize', titleSize, 'FontWeight', 'bold');
h = colorbar; ylabel(h, powerLabel);
end

function generate_regional_bar_chart(averages, electrodes, cfg, outputDir)
bands = fieldnames(averages);
fmt = get_vis(cfg, 'output_format', 'png');
titleSize = get_vis(cfg, 'font_size_title', 16);
labelSize = get_vis(cfg, 'font_size_labels', 10);
cmap = resolve_colormap(cfg);

regions = get_regions(cfg);
regionNames = fieldnames(regions);
nRegions = length(regionNames);
nBands = length(bands);


regionMeans = zeros(nBands, nRegions);
for bi = 1:nBands
    data = averages.(bands{bi});
    for ri = 1:nRegions
        rElectrodes = regions.(regionNames{ri});
        if iscell(rElectrodes)
            indices = find(ismember(electrodes, rElectrodes));
        else
            indices = find(ismember(electrodes, cellstr(rElectrodes)));
        end
        if ~isempty(indices)
            regionMeans(bi, ri) = mean(data(indices), 'omitnan');
        end
    end
end


hFig = figure('Color', 'w', 'Position', [100, 100, 900, 600], 'Visible', 'off');
hBar = bar(regionMeans');

colorIndices = round(linspace(1, size(cmap,1), nBands));
for k = 1:nBands
    hBar(k).FaceColor = cmap(colorIndices(k), :);
    hBar(k).EdgeColor = 'w';
    hBar(k).LineWidth = 0.5;
end

set(gca, 'XTickLabel', regionNames, 'FontSize', labelSize);
xlabel('Brain Region', 'FontSize', labelSize);
ylabel(strrep(get_vis(cfg, 'power_label', 'Absolute Power (µV²)'), 'Absolute', 'Mean Absolute'), 'FontSize', labelSize);
title('Regional Power Distribution', 'FontSize', titleSize, 'FontWeight', 'bold');

bandLabels = cellfun(@(b) [upper(b(1)) lower(b(2:end))], bands, 'UniformOutput', false);
legend(bandLabels, 'FontSize', labelSize, 'Location', 'northwest');
grid on; set(gca, 'GridAlpha', 0.3);


for k = 1:nBands
    xPos = hBar(k).XEndPoints;
    yPos = hBar(k).YEndPoints;
    for vi = 1:length(yPos)
        text(xPos(vi), yPos(vi) + 1, sprintf('%.1f', yPos(vi)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', labelSize - 2);
    end
end

save_figure(hFig, outputDir, 'regional_bar_chart_eeglab', fmt);
end

function generate_heatmap(averages, electrodes, cfg, outputDir)
bands = fieldnames(averages);
fmt = get_vis(cfg, 'output_format', 'png');
titleSize = get_vis(cfg, 'font_size_title', 16);
labelSize = get_vis(cfg, 'font_size_labels', 10);
cmap = resolve_colormap(cfg);


nE = length(electrodes);
nB = length(bands);
matrix = zeros(nE, nB);
for bi = 1:nB
    matrix(:, bi) = averages.(bands{bi})(:);
end

hFig = figure('Color', 'w', 'Position', [100, 100, 500, 800], 'Visible', 'off');
imagesc(matrix);
colormap(cmap);
h = colorbar; ylabel(h, get_vis(cfg, 'power_label', 'Absolute Power (µV²)'));


matMin = min(matrix(:));
matRange = max(matrix(:)) - matMin;
for row = 1:nE
    for col = 1:nB
        val = matrix(row, col);
        normVal = (val - matMin) / matRange;
        if normVal > 0.6
            txtColor = 'w';
        else
            txtColor = 'k';
        end
        text(col, row, sprintf('%.1f', val), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', labelSize - 2, 'Color', txtColor);
    end
end

bandLabels = cellfun(@(b) [upper(b(1)) lower(b(2:end))], bands, 'UniformOutput', false);
set(gca, 'XTick', 1:nB, 'XTickLabel', bandLabels, 'FontSize', labelSize);
set(gca, 'YTick', 1:nE, 'YTickLabel', electrodes, 'FontSize', labelSize);
title('Electrode x Band Power Matrix', 'FontSize', titleSize, 'FontWeight', 'bold');

save_figure(hFig, outputDir, 'electrode_band_heatmap_eeglab', fmt);
end

function print_summary(stats)
bands = fieldnames(stats);
for k = 1:length(bands)
    b = bands{k};
    fprintf('  %s: min=%.3f, max=%.3f, mean=%.3f\n', ...
        b, stats.(b).min, stats.(b).max, stats.(b).mean);
end
end



function cfg = default_config()
cfg.excel_file = '';
cfg.output_dir_matlab = 'topography_plots_eeglab';
cfg.electrodes = {'Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz',...
    'C4','T4','T5','P3','Pz','P4','T6','O1','O2'};
cfg.bands = {'delta','theta','alpha','beta'};
cfg.eeglab_path = '';
cfg.regions = struct(...
    'Frontal',  {{'Fp1','Fp2','F7','F3','Fz','F4','F8'}}, ...
    'Central',  {{'C3','Cz','C4'}}, ...
    'Temporal', {{'T3','T4','T5','T6'}}, ...
    'Parietal', {{'P3','Pz','P4'}}, ...
    'Occipital',{{'O1','O2'}});
cfg.visualization = struct(...
    'colormap', 'viridis', ...
    'dpi', 300, ...
    'output_format', 'png', ...
    'num_contours', 6, ...
    'font_size_title', 16, ...
    'font_size_labels', 10, ...
    'figure_size', [8 8], ...
    'electrode_style', 'labels', ...
    'color_scale', 'auto', ...
    'power_label', 'Absolute Power (µV²)', ...
    'plots', struct('topoplot',true,'topoplot_combined',true,...
    'regional_bar',true,'heatmap',true));
end

function val = get_vis(cfg, key, default)
if isfield(cfg, 'visualization') && isfield(cfg.visualization, key)
    val = cfg.visualization.(key);
else
    val = default;
end
end

function val = get_flag(plotsCfg, key, default)
if isfield(plotsCfg, key)
    val = plotsCfg.(key);
else
    val = default;
end
end

function regions = get_regions(cfg)
if isfield(cfg, 'regions')
    regions = cfg.regions;
else
    d = default_config();
    regions = d.regions;
end
end

function cmap = resolve_colormap(cfg)
cmapName = get_vis(cfg, 'colormap', 'viridis');
% MATLAB doesn't have 'viridis' — map it to 'parula' (equivalent)
if strcmpi(cmapName, 'viridis')
    cmapName = 'parula';
end
try
    cmap = feval(cmapName, 256);
catch
    cmap = parula(256);
end
end

function save_figure(hFig, outputDir, name, fmt)
filePath = fullfile(outputDir, sprintf('%s.%s', name, fmt));
switch lower(fmt)
    case 'png'
        saveas(hFig, filePath);
    case 'svg'
        saveas(hFig, filePath);
    case 'pdf'
        exportgraphics(hFig, filePath, 'ContentType', 'vector');
    otherwise
        saveas(hFig, filePath);
end
fprintf('  Saved: %s\n', filePath);
close(hFig);
end

function dataFile = detect_data_file(cfgPath)
if ~isempty(cfgPath)
    dataFile = cfgPath;
    return;
end
files = [dir('*.xlsx'); dir('*.csv'); dir('*.txt')];
if length(files) == 1
    dataFile = files(1).name;
    fprintf('Auto-detected data file: %s\n', dataFile);
elseif isempty(files)
    error(['No data file found. Place an .xlsx, .csv, or .txt file ' ...
        'in the working directory, or set ''excel_file'' in config.json.']);
else
    names = strjoin({files.name}, ', ');
    error('Multiple data files found: %s. Set ''excel_file'' in config.json to specify which one.', names);
end
end
