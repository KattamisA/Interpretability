clear all
close all

img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator','knife','rope'];
confs = [0.988483, 0.999995, 0.457370, 0.966003, 0.999038, 0.997176, 0.852632, 0.404166, 0.609248, 0.892175, 0.971515, 0.995673, 0.541367, 0.906055, 0.992315, 0.966639, 1.000000, 0.871618, 0.426503, 0.956077];
eps = 1:100;
index = round(max(1.25*eps,eps+4));

figure
common = 'jsma/%s.txt';
Average = zeros(126,20);
Least = zeros(126,20);
for i=1:size(img_names,2)
    path = sprintf(common,img_names(i));
    s = load(path);
    Average(2:end,i) = s(:,2);
    Least(2:end,i) = s(:,3);
end
Averaged = mean(Average,2);
Least = mean(Least,2);
Averaged(1) = sum(confs)/20;
plot(0:100, [Averaged([1,index]), Least([1,index])], 'linewidth', 1)
grid on
xlabel('Adversarial strength, \epsilon')
ylabel('Classification confidence')
legend('Original class','Least Likely Class')

common = '%s/%s.txt';
methods = ['fgsm','bi','llci','jsma'];
for j = 1:4
    Average = zeros(126,20);
    Least = zeros(126,20);
    for i=1:size(img_names,2)
        path = sprintf(common,methods(j),img_names(i));
        s = load(path);
        Average(2:end,i) = s(:,2);
        Least(2:end,i) = s(:,3);
    end
    Averaged(:,j) = mean(Average,2);
    Least(:,j) = mean(Least,2);
    Averaged(1,j) = sum(confs)/20;
end
