%%
s =genpath('../FOCO_lab/CELL_ID');
addpath(s);

%%

fold = 'data/NP_FOCO_cropped';

folders = dir(fold);

for i = 1:size(folders)
    folder = folders(i).name;
    if ~startsWith(folder, '20')
        continue
    end
    
    files = dir(strcat(fold, '/', folder));
    for j = 1:size(files)
        if endsWith(files(j).name, 'ome.mat')
            name = files(j).name
            imfile = strcat(fold, '/', folder, '/', files(j).name);
        end
    end
    
    NP_image = DataHandling.NeuroPALImage;


    [data, info, prefs, worm, mp, neurons, np_file, id_file] = NP_image.open(imfile);
    
    data_RGBW = double(data(:,:,:, prefs.RGBW));
    
    mat2np(data_RGBW, strcat('data/NP_FOCO_cropped/', name, '.pkl'), 'float64')
    
end