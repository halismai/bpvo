I=rgb2gray(imread('~/data/NewTsukubaStereoDataset/illumination/fluorescent/left/tsukuba_fluorescent_L_00001.png'));

sigma_bp = 1.618;
sigma_ct = 0.75;

b = bitplanes(I, sigma_bp, sigma_ct);


load('../build/C0');

