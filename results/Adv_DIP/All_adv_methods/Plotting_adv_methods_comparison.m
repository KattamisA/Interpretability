clear all
close all

t = 0:100:5000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
eps = [1, 5, 25, 100];



Average = zeros(51,4);
Max = zeros(20,4);
Maxconfs = zeros(20,4);
Std = zeros(51,4);
Std = zeros(51, 4);

k=4;

%% Least-likely class iterative
common = 'LLCI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for i=1:size(img_names,2)
    path = sprintf(common,eps(k),img_names(i));
    s = load(path);
    confidences(:,i) = s(1:51,1);
end
Average(:,3) = smooth(mean(confidences,2),1);
Std(:,3) = std(confidences,0,2);
[M, I] = max(confidences(:,:));
Max(:,3) = I*100;
[M2, I2] = max(Average(:,3));
Maxconfs(:,3) = confidences(I2,:);


%% Basic Iterative
common = 'BI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for i=1:size(img_names,2)
    path = sprintf(common,eps(k),img_names(i));
    s = load(path);
    confidences(:,i) = s(1:51,1);
end
Average(:,2) = smooth(mean(confidences,2),1);
Std(:,2) = std(confidences,0,2);
[M, I] = max(confidences(:,:));
Max(:,2) = I*100;
[M2, I2] = max(Average(:,2));
Maxconfs(:,2) = confidences(I2,:);



%% FGSM
common = 'FGSM_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for i=1:size(img_names,2)
    path = sprintf(common,eps(k),img_names(i));
    s = load(path);
    confidences(:,i) = s(1:51,1);
end
Average(:,1) = smooth(mean(confidences,2),1);
Std(:,1) = std(confidences,0,2);
[M, I] = max(confidences(:,:));
Max(:,1) = I*100;
[M2, I2] = max(Average(:,1));
Maxconfs(:,1) = confidences(I2,:);



%% JSMA-SG
common = 'JSMA_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for i=1:size(img_names,2)
    path = sprintf(common,eps(k),img_names(i));
    s = load(path);
    confidences(:,i) = s(1:51,1);
end
Average(:,4) = smooth(mean(confidences,2),1);
Std(:,4) = std(confidences,0,2);
[M, I] = max(confidences(:,:));
Max(:,4) = I*100;
[M2, I2] = max(Average(:,4));
Maxconfs(:,4) = confidences(I2,:);

figure
plot(t,Std, 'linewidth',1.5)
grid on
ylim([0 0.5])
xlabel('DIP iterations')
% ylabel('True Class Confidence')
ylabel('True Class Confidence Standard Deviation')
legend('FGSM','BI','LLCI', 'JSMA-SG')

% figure
% for q = 1:4
% j = 2*q -1;
% [M,I] = max(Average(:,q));
% hold on
% plot([max(Maxconfs(:,q)),min(Maxconfs(:,q))],[j,j],'k','LineWidth',1,'HandleVisibility','off')
% plot([M-1*Std(I,q),M + 1*Std(I,q)],[j,j],'LineWidth',10)
% plot(Maxconfs(:,q),j,'xk','MarkerSize',8,'HandleVisibility','off')
% end
% xlim([0,2])
% ylim([0,8])
% set(gca,'ytick',[])
% xlabel('True Class Confidence')
% grid on
% 
% 
% for q = 1:4
%     figure
%     histogram(Max(:,q),[-50:100:2050])
%     ylim([0 10])
%     grid on 
%     xlabel('DIP iterations')
%     ylabel('# of Occurances')
% end
