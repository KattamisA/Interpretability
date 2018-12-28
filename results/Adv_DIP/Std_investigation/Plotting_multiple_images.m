% clear all
% close all

%% Load and plot confidences
t=0:100:10000;
Confidence = zeros(101,4);
hold on
std = [1, 2, 4, 8, 16, 32];
common = 'EntropySGD/%s_%d_Normalised.txt';
call = '%s_%d_Normalised';
img_names = ["panda", "F16_GT", "monkey",'zebra_GT'];%,'goldfish'];%,'whale','dolphin','spider','labrador','snake','flamingo_animal','canoe','car_wheel','fountain','football_helmet','hourglass','refrigirator'];
Averaged= zeros(101,6);

for j=1:6
    for i=1:4
        path = sprintf(common,img_names(i),std(j));
%         calling = sprintf(call,img_names(i),std(j));
        s = load(path);
        Confidence(:,i) = s(:,1);       
    end
    Averaged(:,j) = smooth(mean(Confidence,2));
end

plot(0:100:10000,Averaged)
legend('show')


% plot(t,Confidence)
% legend('Panda','Peacock','F16 GT','Monkey','Zebra GT','Goldfish')