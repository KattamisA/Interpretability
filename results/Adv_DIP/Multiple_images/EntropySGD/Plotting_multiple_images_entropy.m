% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,18);
hold on

common = '%s/Normalised.txt';
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator'];
for i=1:18
    path = sprintf(common,img_names(i));
    load(path) 
    Confidence(:,i) = Normalised(:,1);
end
plot(t,mean(Confidence,2))

% common = '%s/Confidences.txt';
% img_names = ["panda","peacock","F16_GT","monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator'];
% for i=1:18
%     path = sprintf(common,img_names(i));
%     load(path) 
%     Confidence(:,i) = Confidences(:,1);
% end
% plot(t,mean(Confidence,2))

% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')