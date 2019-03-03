clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

hold on
common = '%s_fgsm.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(:,i) = smooth(s(1:51,2),1);
end
Average = mean(Average,2);
plot(1:2:101, Average, 'linewidth', 1)

common = '%s_bi.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(:,i) = smooth(s(1:51,2),1);
end
Average = mean(Average,2);
plot(1:2:101, Average, 'linewidth', 1)

common = '%s_llci.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(:,i) = smooth(s(1:51,2),1);
end
Average = mean(Average,2);
plot(1:2:101, Average, 'linewidth', 1)

grid on
legend('FGSM', 'BI', 'LLCI')

q = 1:100;
qs = round(max(1.25*q,q+4));