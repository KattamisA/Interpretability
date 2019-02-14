clear all
% close all

%% Load and plot confidences
t=0:100:5000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
             
Confidence = zeros(51,size(img_names,2));
Normalise = zeros(51,size(img_names,2));
hold on

%% Calculating normalised

hold on
common = 'FGSM_eps%d/%s_Normalised.txt';
eps = [100];
for j = 1:1
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        Normalise(:,i) = s(:,1);
        Average(:,j ) = mean(Normalise,2);
        Std(:,j) = std(Normalise,0,2);
    end
end

plot(t,Average,'LineWidth',1.5)
ylim([0 1])
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
grid on
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
figure
hold on
common = 'BI_eps%d/%s_Normalised.txt';
for j = 1:1
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        Normalise(:,i) = s(:,1);
        Average(:,j) = mean(Normalise,2);
        Std(:,j) = std(Normalise,0,2);
    end
end

plot(t,Average,'LineWidth',1.5)
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
grid on
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
ylim([0 1])

figure
hold on
common = 'LLCI_eps%d/%s_Normalised.txt';
for j = 1:1
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        Normalise(:,i) = s(:,1);
        Average(:,j) = mean(Normalise,2);
        Std(:,j) = std(Normalise,0,2);
    end
end

plot(t,Average,'LineWidth',1.5)
plot(t',[Average+Std, Average-Std],'--b','LineWidth',0.2,'HandleVisibility','off')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
grid on
% ylim([0 1])
%% Calculating Unnormalized
% common = 'EntropySGD/%s_Confidences.txt';
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     q = load(path);
%     Confidence(:,i) = q(:,1);
% end

%% Plotting
% figure
% plot(t,mean(Confidence,2))
hold on
legend('FGSM','BI','LLCI')
% plot(t,Std,'LineWidth',1.5)
% plot(t',[Average+Std, Average-Std],'--r','LineWidth',0.2,'HandleVisibility','off')
% legend('Adam optimizer - average')
xlabel('DIP iterations')
ylabel('Averaged Class Confidence')
% legend('Epsilon = 1','Epsilon = 5', 'Epsilon = 25', 'Epsilon = 100')
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