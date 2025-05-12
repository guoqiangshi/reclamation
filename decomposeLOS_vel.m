% Decompose LOS velocity to E, N, U 

% Load input data
[LOS_ASC,LOS_ASCInfo] = readgeoraster('TSX_LOS_A.tif');  % Ascending LOS vel data
LOS_ASC = double(LOS_ASC);
[LOS_DSC,LOS_DSCInfo] = readgeoraster('TSX_LOS_D.tif');  % Descending LOS vel data
LOS_DSC = double(LOS_DSC);
[ASPECT, aspInfo] = readgeoraster('aspect.tif');      % Aspect
ASPECT = double(ASPECT);

% Processing of nodata values
ASPECT(ASPECT == ASPECT1) = NaN; % ASPECT1 is the nodata value
LOS_ASC(LOS_ASC == -inf) = NaN;
LOS_DSC(LOS_DSC == -inf) = NaN;
%%
% Create validity masks for each dataset
mask_ASC = ~isnan(LOS_ASC);  
mask_DSC = ~isnan(LOS_DSC);  
mask_ASPECT = ~isnan(ASPECT);

% Combined mask where all datasets have valid values
final_mask = mask_ASC & mask_DSC & mask_ASPECT;

% Apply mask to all datasets (set invalid pixels to NaN)
LOS_ASC(~final_mask) = NaN;
LOS_DSC(~final_mask) = NaN; 
ASPECT(~final_mask) = NaN;

%% 
% Define sensor geometry parameters (TSX)
tha_A=pi*33.85/180;  %% ASC incidence angle
tha_D=pi*22/180;  %% DSC incidence angle
alph_A=pi*(-10.75)/180;  %% ASC azumuth angle
alph_D=pi*(-169.18)/180;  %% DSC azumuth angle

%% 
% (Sentinel)
tha_A=pi*39.14/180;  %% ASC incidence angle
tha_D=pi*34.92/180;  %% DSC incidence angle
alph_A=pi*(-10.04)/180;  %% ASC azumuth angle
alph_D=pi*(-169.48)/180;  %% DSC azumuth angle
%%
% Initialize output matrices
dE_matrix = nan(size(LOS_ASC));
dN_matrix = nan(size(LOS_ASC)); 
dV_matrix = nan(size(LOS_ASC));

%
for i = 1:size(LOS_ASC, 1)
    for j = 1:size(LOS_ASC, 2)
        if final_mask(i, j)
            %Get pixel values
            LOS_A_ij = LOS_ASC(i, j);
            LOS_D_ij = LOS_DSC(i, j);
            aspect_ij = deg2rad(ASPECT(i, j));

            % Calculate coefficients for decomposition
            B_asc = sin(tha_A)*sin(alph_A) - sin(tha_A)*cos(alph_A)*tan(aspect_ij);
            C_asc = cos(tha_A);
    
            B_dsc = sin(tha_D)*sin(alph_D) - sin(tha_D)*cos(alph_D)*tan(aspect_ij);
            C_dsc = cos(tha_D);

            A = [B_asc, C_asc; B_dsc, C_dsc];
            b = [LOS_A_ij; LOS_D_ij];
            x = A \ b;

            % Store results
            dN_matrix(i, j) = x(1);
            dV_matrix(i, j) = x(2);
            dE_matrix(i, j) = dN_matrix(i, j) * tan(aspect_ij);
        end
    end
end

% Save results as GeoTIFF files
geotiffwrite('tsx_v_n.tif', dN_matrix, LOS_ASCInfo);
geotiffwrite('tsx_v_e.tif', dE_matrix, LOS_ASCInfo);
geotiffwrite('tsx_v_u.tif', dV_matrix, LOS_ASCInfo);