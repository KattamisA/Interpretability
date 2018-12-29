% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Normalise = zeros(101,size(img_names,2));
hold on

common = 'EntropySGD/%s/Normalised.txt';

for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    load(path)
    Normalise(:,i) = Normalised(:,1);
end
% 
% plot(t,mean(Normalise,2))
% title("Normalized class confidence (based on initial confidence)")
% xlabel('DIP iterations')
% ylabel('Class Confidence')

common = '%s/Confidences.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    load(path) 
    Confidence(:,i) = Confidences(:,1);
end
Std = std(Normalise,0,2);
Average = mean(Normalise,2);
% % figure
% % plot(t,mean(Confidence,2))
% hold on
% plot(t,Average)
% hold on
% plot(t,[Average-Std, Average+Std],'--')
% % legend('Unormalised Confidences','Normalised Confidences')
% xlabel('DIP iterations')
% ylabel('Averaged Class Confidence')
% 
% figure
% plot(t,Normalise)
% xlabel('DIP iterations')
% ylabel('Class Confidence')
% 
% figure
% plot(t,Confidence)
% xlabel('DIP iterations')
% ylabel('Class Confidence')
plot(t,Std)

% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')