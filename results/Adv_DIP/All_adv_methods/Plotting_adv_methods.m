clear all
% close all

t = 0:100:5000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
eps = [1, 5, 25, 100];

k=4;
common = 'LLCI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
Average2 = zeros(51,4);
Max = zeros(20,4);
Std = zeros(4, 51);
Std2 = zeros(51, 44);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
    Std(j,:) = std(confidences,0,2);
end
Average2(:,3) = Average(k,:);
Std2(:,3) = Std(k,:);
[M, I] = max(confidences(:,k));
Max(i,1) = I*100;
% figure
% plot(t,Average, 'LineWidth', 1.5)
% grid on
% ylim([0 1.2])
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'BI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
    Std(j,:) = std(confidences,0,2);
end
Average2(:,2) = Average(k,:);
Std2(:,2) = Std(k,:);
[M, I] = max(confidences(:,k));
Max(i,2) = I*100;
% figure
% plot(t,Average, 'LineWidth', 1.5)
% grid on
% ylim([0 1.2])
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'FGSM_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
    Std(j,:) = std(confidences,0,2);
end
Average2(:,1) = Average(k,:);
Std2(:,1) = Std(k,:);
[M, I] = max(confidences(:,k));
Max(i,3) = I*100;
% figure
% plot(t,Average, 'LineWidth', 1.5)
% grid on
% ylim([0 1.2])
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')

common = 'JSMA_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
    Std(j,:) = std(confidences,0,2);
end
Average2(:,4) = Average(k,:);
Std2(:,4) = Std(k,:);
[M, I] = max(confidences(:,k));
Max(:,4) = I*100;
% figure
% plot(t,Average, 'LineWidth', 1.5)
% grid on
% ylim([0 1.2])
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')

% figure
% plot(t,Average2, 'linewidth',1.5)
% grid on
% ylim([0 1.2])
% xlabel('DIP iterations')
% ylabel('True Class Confidence')
% legend('FGSM','BI','LLCI', 'JSMA-SG')

figure
for q = 1:4
j = 2*q -1;
[M,I] = max(Average2(:,q));
% plot([max(Max(q,:)),min(Max(q,:))],[j,j],'k','LineWidth',1,'HandleVisibility','off')
% plot([M-1*Std(I,q),M + 1*Std(I,q)],[j,j],'LineWidth',10)
plot(Max(:,q),j,'xk','MarkerSize',8,'HandleVisibility','off')
end
% xlim([0,2])
ylim([0,8])
set(gca,'ytick',[])
xlabel('Average Class Confidence')






