clear all
close all


%% Create containers
Confidence = zeros(100,6);
Std_conf = zeros(101,5);
simple_FGSM_conf = zeros(101,3);
f = 7;
% %% Load variables
% load 'LLCI_eps100/LLCI_eps100_full.txt'
% load 'LLCI_eps25/LLCI_eps25_full.txt'
% load 'LLCI_eps5/LLCI_eps5_full.txt'
% 
% load 'FGSM_eps100/FGSM_eps100_full.txt'
% load 'FGSM_eps25/FGSM_eps25_full.txt'
% load 'FGSM_eps5/FGSM_eps5_full.txt'
% 
% load 'BI_eps100/BI_eps100_full.txt'
% load 'BI_eps25/BI_eps25_full.txt'
% load 'BI_eps5/BI_eps5_full.txt'
% 
% load 'Simple_FGSM_5/Simple_FGSM_5_full.txt'
% load 'Simple_FGSM_25/Simple_FGSM_25_full.txt'
% load 'Simple_FGSM_100/Simple_FGSM_100_full.txt'
% 
% simple_FGSM_conf(:,1) = smooth(Simple_FGSM_5_full(:,1),f);
% simple_FGSM_conf(:,2) = smooth(Simple_FGSM_25_full(:,1),f);
% simple_FGSM_conf(:,3) = smooth(Simple_FGSM_100_full(:,1),f);
% 
% load 'Complex_LLCI_eps100/Complex_LLCI_eps100_full.txt'


load 'Std_complex_1-64/Confidences.txt'
Std_conf(:,1) = smooth(Confidences(:,1),f);
load 'Std_complex_2-64/Confidences.txt'
Std_conf(:,2) = smooth(Confidences(:,1),f);
load 'Std_complex_4-64/Confidences.txt'
Std_conf(:,3) = smooth(Confidences(:,1),f);
load 'Std_complex_8-64/Confidences.txt'
Std_conf(:,4) = smooth(Confidences(:,1),f);
load 'Std_complex_16-64/Confidences.txt'
Std_conf(:,5) = smooth(Confidences(:,1),f);
load 'Std_complex_32-64/Confidences.txt'
Std_conf(:,6) = smooth(Confidences(:,1),f);
% 
% load 'ID_complex_2/Confidences.txt'
% Confidence(:,1) = smooth(Confidences(:,1),f);
% load 'ID_complex_4/Confidences.txt'
% Confidence(:,2) = smooth(Confidences(:,1),f);
% load 'ID_complex_8/Confidences.txt'
% Confidence(:,3) = smooth(Confidences(:,1),f);
% load 'ID_complex_16/Confidences.txt'
% Confidence(:,4) = smooth(Confidences(:,1),f);
% load 'ID_complex_32/Confidences.txt'
% Confidence(:,5) = smooth(Confidences(:,1),f);
% load 'ID_complex_64/Confidences.txt'
% Confidence(:,6) = smooth(Confidences(:,1),f);
% t=0:100:9900;
% plot(t,Confidence);
% 
% legend('Input depth = 2','Input depth = 4','Input depth = 8','Input depth = 16','Input depth = 32','Input depth = 64');
% xlabel('DIP iterations');
% ylabel('Confidence of true class');
% 
% figure
% plot(t,reshape(smooth(Confidences(:,[1,2]),f),100,2))
% xlabel('DIP iterations');
% ylabel('Class Confidence');

figure
plot(0:100:10000,Std_conf)
xlabel('DIP iterations');
ylabel('Confidence of true class');
legend('show')
% figure
% plot(0:100:10000,simple_FGSM_conf)
% xlabel('DIP iterations');
% ylabel('Confidence of true class');

% figure
% plot(0:100:9100,Complex_LLCI_eps100_full(:,[1,2]))
%% Plotting all the LLCI true classes confidences

% True_class_confidence = [LLCI_eps5_full(:,1), LLCI_eps25_full(:,1), LLCI_eps100_full(:,1)];
% plot(0:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Least Likely Class Iterative Method')
% grid on

%% Plotting all the FGSM true classes confidences
% figure
% True_class_confidence = [FGSM_eps5_full(:,1), FGSM_eps25_full(:,1), FGSM_eps100_full(:,1)];
% plot(0:100:9900,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Fast Gradient Sign Method')
% grid on

%% Plotting all the BI true classes confidences
%  
% True_class_confidence = [BI_eps5_full(:,1), BI_eps25_full(:,1), BI_eps100_full(:,1)];
% plot(0:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Basic Iterative Method')
% grid on

%% Comparing FGSM and LLCIfor true class

% t = 100:100:10000;

% subplot(1,3,1) % first subplot - eps = 5
% plot(t,[FGSM_eps5_full(:,1),BI_eps5_full(2:end,1), LLCI_eps5_full(2:end,1)])
% % xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% % legend('Fast Gradient Sign Method','Least Likely Class Iterative')
% grid on
% title('\epsilon = 5')

% subplot(1,3,2) % first subplot - eps = 25
% plot(t,[FGSM_eps25_full(:,1),BI_eps25_full(2:end,1), LLCI_eps25_full(2:end,1)])
% xlabel('DIP Iterations')
% % ylabel('True Class Confidence')
% % legend('Fast Gradient Sign Method','Least Likely Class Iterative')
% grid on
% title('\epsilon = 25')

% subplot(1,3,3) % first subplot - eps = 100
% plot(t,[FGSM_eps100_full(:,1),BI_eps100_full(2:end,1), LLCI_eps100_full(2:end,1)])
% % xlabel('DIP Iterations')
% % ylabel('True Class Confidence')
% legend('Fast Gradient Sign Method','Basic Iterative','Least Likely Class Iterative')
% grid on
% title('\epsilon = 100')

%% Complex architecture for LLCI eps=100, plotting first two classes

% t=0:100:9200 
% plot(t,Complex_LLCI_eps100_full(:,1:2))


%% Plotting the top 4 classes for FGSM
% t = 100:100:10000;
% 
% plot(t,FGSM_eps100_full(:,[1,2:4]))
% xlabel('DIP Iterations')
% ylabel('Class Confidence')
% legend('True class','Incorrect class(1)','Incorrect class(2)','Incorrect class(3)')
% title('Fast Gradient Sign Method with \epsilon = 100')
% figure
% plot(t,FGSM_eps25_full(:,[1,2:4]))
% xlabel('DIP Iterations')
% ylabel('Class Confidence')
% legend('True class','Incorrect class(1)','Incorrect class(2)','Incorrect class(3)')
% title('Fast Gradient Sign Method with \epsilon = 25')
% figure
% plot(t,FGSM_eps5_full(:,[1,2:4]))
% xlabel('DIP Iterations')
% ylabel('Class Confidence')
% legend('True class','Incorrect class(1)','Incorrect class(2)','Incorrect class(3)')
% title('Fast Gradient Sign Method with \epsilon = 5')
