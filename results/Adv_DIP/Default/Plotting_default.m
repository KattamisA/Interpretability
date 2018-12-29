% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];

Confidence = zeros(101,size(img_names,2));
Normalise = zeros(101,size(img_names,2));
hold on

common = 'EntropySGD/%s_Normalised.txt';

for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Normalise(:,i) = s(:,1);
end
% 
% plot(t,mean(Normalise,2))
% title("Normalized class confidence (based on initial confidence)")
% xlabel('DIP iterations')
% ylabel('Class Confidence')

common = 'EntropySGD/%s_Confidences.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    q = load(path);
    Confidence(:,i) = q(:,1);
end

% figure
% plot(t,mean(Confidence,2))
hold on
plot(t,mean(Normalise,2))
% legend('Unormalised Confidences','Normalised Confidences')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
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

% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')