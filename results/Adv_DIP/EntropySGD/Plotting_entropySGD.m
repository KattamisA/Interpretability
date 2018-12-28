clear all
close all

%% Load variables
t=0:100:10000;
tests = ['0.01','0.05','0.1','0.5','1','2','5','10','20','50','100'];

common = 'LR_complex_%d/Confidences.txt';

for i=2:8
    path = sprintf(common,i);
    load(path)
    Normalise(:,i+1) = smooth(Confidences(:,1),5);
%     legend([tests(i+1)])
end
plot(t,Normalise)
legend('0.1','0.5','1','2','5','10','20')
grid on
xlabel('DIP iterations')
ylabel('True Class Confidence')

% hold on
% load 'EntropySGD/Std_complex_1-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% load 'EntropySGD/Std_complex_2-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% load 'EntropySGD/Std_complex_4-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% load 'EntropySGD/Std_complex_8-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% load 'EntropySGD/Std_complex_16-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% load 'EntropySGD/Std_complex_32-64/Confidences.txt'
% plot(0:100:10000,Confidences(:,1))
% legend('show')

