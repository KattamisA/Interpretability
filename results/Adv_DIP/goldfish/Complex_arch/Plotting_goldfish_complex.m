clear all
close all

%% Create variables
t=0:100:10000;
Smoothing_factor = 7;

%% Load variables

% Varying noise standard deviation
Std_conf = zeros(101,6);
load 'Std_complex_1-64/Confidences.txt'
Std_conf(:,1) = smooth(Confidences(:,1),Smoothing_factor);
load 'Std_complex_2-64/Confidences.txt'
Std_conf(:,2) = smooth(Confidences(:,1),Smoothing_factor);
load 'Std_complex_4-64/Confidences.txt'
Std_conf(:,3) = smooth(Confidences(:,1),Smoothing_factor);
load 'Std_complex_8-64/Confidences.txt'
Std_conf(:,4) = smooth(Confidences(:,1),Smoothing_factor);
load 'Std_complex_16-64/Confidences.txt'
Std_conf(:,5) = smooth(Confidences(:,1),Smoothing_factor);
load 'Std_complex_32-64/Confidences.txt'
Std_conf(:,6) = smooth(Confidences(:,1),Smoothing_factor);

%Varying input depth
ID_conf = zeros(101,6);
load 'ID_complex_2/Confidences.txt'
ID_conf(:,1) = smooth(Confidences(:,1),Smoothing_factor);
load 'ID_complex_4/Confidences.txt'
ID_conf(:,2) = smooth(Confidences(:,1),Smoothing_factor);
load 'ID_complex_8/Confidences.txt'
ID_conf(:,3) = smooth(Confidences(:,1),Smoothing_factor);
load 'ID_complex_16/Confidences.txt'
ID_conf(:,4) = smooth(Confidences(:,1),Smoothing_factor);
load 'ID_complex_32/Confidences.txt'
ID_conf(:,5) = smooth(Confidences(:,1),Smoothing_factor);
load 'ID_complex_64/Confidences.txt'
ID_conf(1:100,6) = smooth(Confidences(:,1),Smoothing_factor);

%% Plotting

plot(t,ID_conf)
legend('Input depth = 2','Input depth = 4','Input depth = 8','Input depth = 16','Input depth = 32','Input depth = 64');
xlabel('DIP iterations');
ylabel('Confidence of true class');

figure
plot(t,Std_conf)
xlabel('DIP iterations');
ylabel('Confidence of true class');
legend('Std = 1/64','Std = 2/64','Std = 4/64','Std = 8/64','Std = 16/64','Std = 32/64')
