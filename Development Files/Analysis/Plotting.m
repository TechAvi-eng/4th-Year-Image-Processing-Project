clear all
close all
clc

% Plotting of analysis data
load('dataExtracted')

% Extract Cell Type IDs (Column 2) and Number of Cells (Column 4)
cellTypeIDs = dataExtracted(:, 2);
time = dataExtracted(:, 3);
numCells = dataExtracted(:, 4);
avgArea = dataExtracted(:, 5);
avgPer = dataExtracted(:, 6);

cellNames = {'A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3'};
colors = lines(8);

figure (1); 
hold on;

for i = 1:8
    idx = (cellTypeIDs == i); % logical array takes only cell types = i for plotting
    scatter(time(idx), avgArea(idx), 50, colors(i, :), 'filled', 'DisplayName', cellNames{i});
end

xlabel('Time', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Average Cell Area', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Scatter Plot of Average Area vs Time', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('Location', 'best', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
hold off;


figure (2); 
hold on;

for i = 1:8
    idx = (cellTypeIDs == i); % logical array takes only cell types = i for plotting
    scatter(time(idx), numCells(idx), 50, colors(i, :), 'filled', 'DisplayName', cellNames{i});
end

xlabel('Time', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Number of Cells', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Scatter Plot of Number of Cells vs Time', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('Location', 'best', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
hold off;



figure (3); 
hold on;

for i = 1:8
    idx = (cellTypeIDs == i); % logical array takes only cell types = i for plotting
    scatter(time(idx), avgPer(idx), 50, colors(i, :), 'filled', 'DisplayName', cellNames{i});
end

xlabel('Time', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Average Cell Perimeter', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Scatter Plot of Average Perimeter vs Time', 'FontSize', 14, 'FontName', 'Times New Roman');
legend('Location', 'best', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
hold off;

% % Get unique Cell Type IDs and their corresponding total counts
% uniqueTypes = unique(cellTypeIDs); % Find unique cell types
% numCellsPerType = zeros(size(uniqueTypes)); % Initialize storage
% 
% % Sum up the number of cells for each unique cell type
% for i = 1:length(uniqueTypes)
%     numCellsPerType(i) = sum(numCells(cellTypeIDs == uniqueTypes(i)));
% end
% 
% % Create bar graph
% bar(uniqueTypes, numCellsPerType);
% 
% % Add labels and title
% xlabel('Cell Type ID');
% ylabel('Number of Cells');
% title('Number of Cells per Cell Type');
% 
% % Improve x-axis display
% xticks(uniqueTypes); % Set x-ticks to unique cell types
% grid on; % Add grid for better readability