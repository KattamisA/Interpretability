clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = [ "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador'];
Confidence = zeros(101,size(img_names,2));
Normalise = zeros(101,size(img_names,2));
hold on

%% Calculating normalised
common = 'LLCI_eps100/%s_Normalised.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Normalise(:,i) = smooth(s(:,1),3);
end

%% Calculating Unnormalized
% common = 'EntropySGD/%s_Confidences.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     q = load(path);
%     Confidence(:,i) = q(:,1);
% end

%% Plotting
Average = mean(Normalise,2);
Std = std(Normalise,0,2);
% figure
% plot(t,mean(Confidence,2))
hold on
plot(0:100:10000,Average(1:101,:),'LineWidth',1.5)
% plot(t',[Average+Std, Average-Std],'--r','LineWidth',0.2,'HandleVisibility','off')
% legend('Adam optimizer - average')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')

%% Plotting all the lines 
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