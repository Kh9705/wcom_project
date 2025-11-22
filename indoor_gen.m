% Setup the Scenario (Same as your original code)
mapFileName = "conferenceroom.stl";
fc = 5.8e9;
txArray = arrayConfig(Size=[4 1], ElementSpacing=2*(physconst("lightspeed")/fc));
rxArray = arrayConfig(Size=[4 4], ElementSpacing=(physconst("lightspeed")/fc));

% Transmitter is fixed
tx = txsite("cartesian", Antenna=txArray, ...
    AntennaPosition=[-1.46; -1.42; 2.1], TransmitterFrequency=fc);

% Define Ray Tracing Model
pm = propagationModel("raytracing", "CoordinateSystem","cartesian", ...
    "Method","sbr", "MaxNumReflections",2, "SurfaceMaterial","wood");

% --- DATA GENERATION LOOP ---
numSamples = 1000; % Number of random positions to train on
dataMatrix = zeros(numSamples, 4); % Columns: [Rx_X, Rx_Y, Rx_Z, PathLoss]

disp('Starting Data Generation...');

for i = 1:numSamples
    % 1. Randomize Receiver Position (Within room limits)
    % Room approx limits: X[-2, 2], Y[-2, 2], Z[0.8, 1.5]
    randX = (rand * 4) - 2; 
    randY = (rand * 4) - 2;
    randZ = 0.85; % Keep height constant for table level, or use (0.8 + rand*0.5) for 3D
    
    rx = rxsite("cartesian", Antenna=rxArray, ...
        AntennaPosition=[randX; randY; randZ], AntennaAngle=[0;90]);
    
    % 2. Run Ray Tracing
    rays = raytrace(tx, rx, pm);
    
    % 3. Extract Data (Smallest Path Loss represents the dominant signal)
    if ~isempty(rays{1})
        rayData = rays{1};
        minPathLoss = min([rayData.PathLoss]); 
        
        % Store: X, Y, Z, PathLoss
        dataMatrix(i, :) = [randX, randY, randZ, minPathLoss];
    else
        % If no rays found (out of bounds), store NaN to filter later
        dataMatrix(i, :) = [randX, randY, randZ, NaN];
    end
    
    if mod(i, 50) == 0
        fprintf('Processed %d / %d samples\n', i, numSamples);
    end
end

% Remove failed rows (NaN)
dataMatrix = dataMatrix(~isnan(dataMatrix(:,4)), :);

% Save to CSV for Python ML
T = array2table(dataMatrix, 'VariableNames', {'Rx_X', 'Rx_Y', 'Rx_Z', 'PathLoss'});
writetable(T, 'ray_tracing_data.csv');
disp('Data generation complete. Saved to ray_tracing_data.csv');