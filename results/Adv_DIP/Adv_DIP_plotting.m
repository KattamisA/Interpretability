clear all
close all

%% Load variables
% load 'LLCI_eps100_full.txt'
% load 'LLCI_eps25_full.txt'
% load 'LLCI_eps5_full.txt'
% 
% load 'FGSM_eps100_full.txt'
% load 'FGSM_eps25_full.txt'
% load 'FGSM_eps5_full.txt'
% 
% load 'BI_eps100_full.txt'
% load 'BI_eps25_full.txt'
% load 'BI_eps5_full.txt'
% 
% load 'Complex_LLCI_eps100_full.txt'
% 
% load 'panda.jpg/Stats.txt'
% Stats1 = Stats;
% load 'peacock.jpg/Stats.txt'
% Stats2 = Stats;
% 
% plot(100:100:10000,[Stats1(:,2),Stats2(:,2)])
hold on
load 'EntropySGD/LR_complex_0/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/LR_complex_1/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/LR_complex_2/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/LR_complex_3/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
load 'EntropySGD/LR_complex_4/Confidences.txt'
plot(0:100:10000,Confidences(:,1))
legend('show')

%% Plotting all the LLCI true classes confidences

% True_class_confidence = [LLCI_eps5_full(:,1), LLCI_eps25_full(:,1), LLCI_eps100_full(:,1)];
% plot(100:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Least Likely Class Iterative Method')
% grid on

%% Plotting all the FGSM true classes confidences
 
% True_class_confidence = [FGSM_eps5_full(:,1), FGSM_eps25_full(:,1), FGSM_eps100_full(:,1)];
% plot(100:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Fast Gradient Sign Method')
% grid on

%% Plotting all the BI true classes confidences
 
% True_class_confidence = [BI_eps5_full(:,1), BI_eps25_full(:,1), BI_eps100_full(:,1)];
% plot(100:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Basic Iterative Method')
% grid on

%% Comparing FGSM and LLCIfor true class

% t = 100:100:10000;

% subplot(1,3,1) % first subplot - eps = 5
% plot(t,[FGSM_eps5_full(:,1),BI_eps5_full(:,1), LLCI_eps5_full(:,1)])
% % xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% % legend('Fast Gradient Sign Method','Least Likely Class Iterative')
% grid on
% title('\epsilon = 5')

% subplot(1,3,2) % first subplot - eps = 25
% plot(t,[FGSM_eps25_full(:,1),BI_eps25_full(:,1), LLCI_eps25_full(:,1)])
% xlabel('DIP Iterations')
% % ylabel('True Class Confidence')
% % legend('Fast Gradient Sign Method','Least Likely Class Iterative')
% grid on
% title('\epsilon = 25')
% 
% % subplot(1,3,3) % first subplot - eps = 100
% plot(t,[FGSM_eps100_full(:,1),BI_eps100_full(:,1), LLCI_eps100_full(:,1)])
% % xlabel('DIP Iterations')
% % ylabel('True Class Confidence')
% legend('Fast Gradient Sign Method','Basic Iterative','Least Likely Class Iterative')
% grid on
% title('\epsilon = 100')


%% Complex architecture for LLCI eps=100, plotting first two classes


% t=0:100:9100;
% 
% plot(t,reshape(smooth(Complex_LLCI_eps100_full(:,1:2),7),92,2))
% xlabel('DIP Iterations')
% ylabel('Class Confidence')
% legend('True class','Incorrect class')


%% Plotting true and second best class for the FGSM

% t = 100:100:10000;
% 
% plot(t,reshape(smooth(FGSM_eps100_full(:,[1,3:6]),7),100,5))
% xlabel('DIP Iterations')
% ylabel('Class Confidence')
% legend('True class','Incorrect class')
