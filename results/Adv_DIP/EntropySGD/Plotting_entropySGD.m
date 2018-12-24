clear all
close all

%% Load variables

hold on
% load 'LR_complex_0/Confidences.txt' % 0.01
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_1/Confidences.txt' % 0.05
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_2/Confidences.txt' % 0.1
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_3/Confidences.txt' % 0.5
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_4/Confidences.txt' % 1
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_5/Confidences.txt' % 2
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_6/Confidences.txt' % 5
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_7/Confidences.txt' % 10
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_8/Confidences.txt' % 20
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_9/Confidences.txt' % 50
% plot(0:100:10000,Confidences(:,1))
% load 'LR_complex_10/Confidences.txt' % 100
% plot(0:100:10000,Confidences(:,1))
legend('show')

hold on
load 'EntropySGD/Std_complex_1-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/Std_complex_2-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/Std_complex_4-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/Std_complex_8-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/Std_complex_16-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/Std_complex_32-64/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
legend('show')

