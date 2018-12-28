% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];
Confidence = zeros(101,size(img_names,2));
hold on
std = [1, 2, 4, 8, 16, 32];
common = 'EntropySGD/%s_%d_Normalised.txt';
call = '%s_%d_Normalised';
Averaged= zeros(101,size(std,2));

for j=1:6
    for i=1:size(img_names,2)
        path = sprintf(common,img_names(i),std(j));
        s = load(path);
        Confidence(:,i) = s(:,1);       
    end
    Averaged(:,j) = smooth(mean(Confidence,2));
end

plot(0:100:10000,Averaged)
xlabel('DIP iterations')
ylabel('Averaged True Class Confidence')
legend('Std = 1/64','Std = 1/32','Std = 1/16','Std = 1/8','Std = 1/4','Std = 1/2')
grid on

% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')