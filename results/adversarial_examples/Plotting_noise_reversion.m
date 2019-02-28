clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = 'LLCI_eps100/%s_stats.txt';


for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(:,i) = s(:,1);
end
