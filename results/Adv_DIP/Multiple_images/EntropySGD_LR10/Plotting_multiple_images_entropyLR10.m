% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator'];

Normalise = zeros(101,size(img_names,2));
Confidence = zeros(101,size(img_names,2));
hold on

common = '%s/Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    load(path) 
    Normalise(:,i) = smooth(Normalised(:,1),5);
end

common = '%s/Confidences.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    load(path) 
    Confidence(:,i) = smooth(Confidences(:,1),5);
end

% plot(t,mean(Confidence,2))
hold on
plot(t,mean(Normalise,2))
xlabel('DIP iterations')
ylabel('Averaged True Class Confidence')
% legend('Unnormalized','Normalized')
grid on


% figure
% plot(t,Normalise)
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% grid on
% 
% figure
% plot(t,Confidence)
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% grid on
% ylim([0,1.4])