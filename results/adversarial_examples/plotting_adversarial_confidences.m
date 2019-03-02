clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

common = '%s_llci.txt';

for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(:,i) = s(1:51,2);
end

% Average = mean(Average,2);

plot(1:2:101, Average)