function [ f ] = DataDensityPlot2( x, y, levels )
    %DATADENSITYPLOT Plot the data density 
    %   Makes a contour map of data density
    %   x, y - data x and y coordinates
    %   levels - number of contours to show
    %
    % By Malcolm Mclean
    %
    
    % Compute data density using the dataDensity function
    map = dataDensity(x, y, 256, 256);
    
    % Normalize the data density values to the range [0, 1]
    map = map - min(min(map));
    map = floor(map ./ max(max(map)) * (levels-1));
    
    % Create a figure
    f = figure();
    
    % Display the data density map as an image
    image(map);
    
    % Set the colormap to grayscale with the specified number of levels
    colormap(gray(levels));
    
    % Set X-axis ticks and labels
    set(gca, 'XTick', [1 256]);
    set(gca, 'XTickLabel', [min(x) max(x)]);
    
    % Set Y-axis ticks and labels
    set(gca, 'YTick', [1 256]);
    set(gca, 'YTickLabel', [min(y) max(y)]);
    
    % Wait for user interaction (UI control)
    uiwait;
end
