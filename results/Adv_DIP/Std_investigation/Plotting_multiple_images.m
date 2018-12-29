clear all
close all

%% Load and plot confidences
t=0:100:10000;
% img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador'];
Confidence = zeros(101,size(img_names,2));
hold on
std1 = [1, 2, 4, 8];
common = 'EntropySGD/%s_%d_256_Normalised.txt';
Averaged= zeros(101,size(std1,2));
Std = zeros(101,size(std1,2));

for j=1:size(std1,2)
    for i=1:size(img_names,2)
        path = sprintf(common,img_names(i),std1(j));
        s = load(path);
        Confidence(:,i) = s(:,1);       
    end
    Averaged(:,j) = smooth(mean(Confidence,2));
    Std(:,j) = std(Confidence,0,2);
end

% plot(0:100:10000,Averaged)
xlabel('DIP iterations')
ylabel('Averaged True Class Confidence')
% legend('Std = 1/64','Std = 1/32','Std = 1/16','Std = 1/8','Std = 1/4','Std = 1/2')
legend('Std = 1/256','Std = 1/128','Std = 1/64','Std = 1/32');
grid on

plot(t,Averaged(:,3))
plot(t,[Averaged(:,3)-Std(:,3), Averaged(:,3)+Std(:,3)],'--')




% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')