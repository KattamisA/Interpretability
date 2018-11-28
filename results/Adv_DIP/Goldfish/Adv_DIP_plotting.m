clear all
close all

%% Load variables
load 'LLCI_eps100/LLCI_eps100_full.txt'
load 'LLCI_eps25/LLCI_eps25_full.txt'
load 'LLCI_eps5/LLCI_eps5_full.txt'

load 'FGSM_eps100/FGSM_eps100_full.txt'
load 'FGSM_eps25/FGSM_eps25_full.txt'
load 'FGSM_eps5/FGSM_eps5_full.txt'

load 'BI_eps100/BI_eps100_full.txt'
load 'BI_eps25/BI_eps25_full.txt'
load 'BI_eps5/BI_eps5_full.txt'

% load 'Complex_LLCI_eps100_full.txt'

%% Plotting all the LLCI true classes confidences

% True_class_confidence = [LLCI_eps5_full(:,1), LLCI_eps25_full(:,1), LLCI_eps100_full(:,1)];
% plot(0:100:10000,True_class_confidence);
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
%  
% True_class_confidence = [BI_eps5_full(:,1), BI_eps25_full(:,1), BI_eps100_full(:,1)];
% plot(0:100:10000,True_class_confidence);
% xlabel('DIP Iterations')
% ylabel('True Class Confidence')
% legend('\epsilon = 5','\epsilon = 25', '\epsilon = 100')
% title('Basic Iterative Method')
% grid on

%% Comparing FGSM and LLCIfor true class

t = 100:100:10000;

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
