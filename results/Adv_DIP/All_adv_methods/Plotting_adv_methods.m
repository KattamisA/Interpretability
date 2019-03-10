clear all
close all

t = 0:100:5000;
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
eps = [1, 5, 25, 100];

common = 'LLCI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
Average2 = zeros(51,4);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
end
Average2(:,4) = Average(4,:);
figure
plot(t,Average, 'LineWidth', 1.5)
grid on
ylim([0 1.2])
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'BI_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
end
Average2(:,3) = Average(4,:);
figure
plot(t,Average, 'LineWidth', 1.5)
grid on
ylim([0 1.2])
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')


common = 'FGSM_eps%d/%s_Normalised.txt';
confidences = zeros(51,20);
for j=1:4
    for i=1:size(img_names,2)
        path = sprintf(common,eps(j),img_names(i));
        s = load(path);
        confidences(:,i) = s(1:51,1);
    end
    Average(j,:) = smooth(mean(confidences,2),1);
end
Average2(:,2) = Average(4,:);
figure
plot(t,Average, 'LineWidth', 1.5)
grid on
ylim([0 1.2])
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')

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
Average2(:,1) = Average(4,:);
figure
plot(t,Average, 'LineWidth', 1.5)
grid on
ylim([0 1.2])
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('Epsilon=1','Epsilon=5','Epsilon=25','Epsilon=100')

figure
plot(t,Average2, 'linewidth',1.5)
grid on
ylim([0 1.2])
xlabel('DIP iterations')
ylabel('True Class Confidence')
legend('JSMA','FGSM','BI','LLCI')