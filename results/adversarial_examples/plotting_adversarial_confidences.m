clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
confs = [0.988483, 0.999995, 0.457370, 0.966003, 0.999038, 0.997176, 0.852632, 0.404166, 0.609248, 0.892175, 0.971515, 0.995673, 0.541367, 0.906055, 0.992315, 0.966639, 1.000000, 0.871618, 0.426503, 0.956077];
eps = 1:100;
index = round(max(1.25*eps,eps+4));

% figure
% common = 'Adversarial_test/jsma/%s.txt';
% Average = zeros(126,20);
% Least = zeros(126,20);
% for i=1:size(img_names,2)
%     path = sprintf(common,img_names(i));
%     s = load(path);
%     Average(2:end,i) = s(:,2);
%     Least(2:end,i) = s(:,3);
% end
% Averaged = mean(Average,2);
% Least = mean(Least,2);
% Averaged(1) = sum(confs)/20;
% plot(0:100, [Averaged([1,index]), Least([1,index])], 'linewidth', 1)
% grid on
% xlabel('Adversarial strength, \epsilon')
% ylabel('Classification confidence')
% legend('Original class','Least Likely Class')

figure
Average = zeros(101,20);
Least2 = zeros(101,20);
common = 'Adversarial_test/fgsm/%s.txt';
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(2:end,i) = s(:,2);
    Least2(2:end,i) = s(:,3);
end
Averaged = smooth(mean(Average,2),7);
Leasts2 = mean(Least2,2);
Averaged(1) = sum(confs)/20;
plot(0:100, Averaged, 'linewidth', 1.5)

hold on
common = 'Adversarial_test/%s/%s.txt';
methods = ["bi", "llci", "jsma"];
Averaged = zeros(126,3);
for j = 1:size(methods,2)
    Average = zeros(126,20);
    Least = zeros(126,20);
    for i=1:size(img_names,2)
        path = sprintf(common, methods(j), img_names(i));
        s = load(path);
        Average(2:end,i) = s(:,2);
        Least(2:end,i) = s(:,3);
    end
    Averaged(:,j) = smooth(mean(Average,2),7);
    Leasts(:,j) = mean(Least,2);
    Averaged(1,j) = sum(confs)/20;
end

plot(0:100, Averaged([1,index],:), 'linewidth', 1.5)
grid on
xlabel('Adversarial strength, \epsilon')
ylabel('Classification confidence')
legend('Fast Gradient Sign Method, FGSM', 'Basic Iterative Method, BI', 'Least Likely Class Iterative Method, LLCI', ' Jacobian-based Saliency Map Approach, JSMA')
xlim([0 105])

figure
plot(0:100, [Leasts2 , Leasts([1,index],:)], 'linewidth', 1.5)
grid on
xlabel('Adversarial strength, \epsilon')
ylabel('Classification confidence')
xlim([0 105])
legend('Fast Gradient Sign Method, FGSM', 'Basic Iterative Method, BI', 'Least Likely Class Iterative Method, LLCI', ' Jacobian-based Saliency Map Approach, JSMA')

