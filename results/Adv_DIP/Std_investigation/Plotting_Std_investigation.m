clear all
% close all

%% Load and plot confidences
t=0:100:10000;
% img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];
img_names = ["panda", "peacock", "F16_GT", "monkey",'zebra_GT','goldfish','whale','dolphin','spider','labrador'];
Confidence = zeros(101,size(img_names,2));
common = 'EntropySGD/%s_%d_256_Normalised.txt';

std1 = [1, 2, 4, 8];
Averaged= zeros(101,size(std1,2));
Std = zeros(101,size(std1,2));

%% Filling in the matrices
for j=1:size(std1,2)
    for i=1:size(img_names,2)
        path = sprintf(common,img_names(i),std1(j));
        s = load(path);
        Confidence(:,i) = s(:,1);       
    end
    Averaged(:,j) = smooth(mean(Confidence,2));
    Std(:,j) = std(Confidence,0,2);
end



%% Plotting
% plot(t,Averaged,'LineWidth',1)
% figure
hold on
plot(0:100:5000,Averaged(1:51,3),'b','LineWidth',1.5)
plot(0:100:5000,[Averaged(1:51,3)-Std(1:51,3), Averaged(1:51,3)+Std(1:51,3)],'--b')
xlabel('DIP iterations')
ylabel('True Class Confidence')
% legend('\sigma_{noise} = 1/64','\sigma_{noise} = 1/32','\sigma_{noise} = 1/16','\sigma_{noise} = 1/8','\sigma_{noise} = 1/4','\sigma_{noise} = 1/2')
% legend('Std = 1/256','Std = 1/128','Std = 1/64','Std = 1/32');
grid on
