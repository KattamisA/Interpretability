clear all;
close all;

%% Plotting for the LCCI
% load 'LCCI_[50_100].txt'
% load 'LLCI_[1_5_10_25].txt'
% LCCI_noisy = zeros(128,6);
% 
% LCCI_noisy(:,5:6) = LCCI__50_100_(:,1:2);
% LCCI_noisy(1:64,1:4) = LLCI__1_5_10_25_(:,1:4);
% 
% plot(1:64,reshape(smooth(LCCI_noisy(1:64,:),5),64,6))
% 
% xlabel('Noise standard deviation')
% ylabel('Confidence of true class')
% legend('\epsilon = 1', '\epsilon = 5', '\epsilon = 10', '\epsilon = 25', '\epsilon = 50', '\epsilon = 100')

%% Plotting for the FGSM
% load FGSM.txt
% 
% plot(1:128,reshape(smooth(FGSM(1:128,:),15),128,6))
% 
% xlabel('Noise standard deviation')
% ylabel('Confidence of true class')
% legend('\epsilon = 1', '\epsilon = 5', '\epsilon = 10', '\epsilon = 25', '\epsilon = 50', '\epsilon = 100')

%% Plotting the samples for FGSM

% load 'FGSM_eps1.txt'
% load 'FGSM_eps5.txt'
% load 'FGSM_eps10.txt'
% load 'FGSM_eps25.txt'
% load 'FGSM_eps50.txt'
% load 'FGSM_eps100.txt'
% 
% subplot(2,3,1)
% plot(1:128,FGSM_eps1,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 1')
% subplot(2,3,2)
% plot(1:128,FGSM_eps5,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 5')
% subplot(2,3,3)
% plot(1:128,FGSM_eps10,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 10')
% subplot(2,3,4)
% plot(1:128,FGSM_eps25,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 25')
% subplot(2,3,5)
% plot(1:128,FGSM_eps50,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 50')
% subplot(2,3,6)
% plot(1:128,FGSM_eps100,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 100')

%% Plotting the samples for LCCI

% load 'LLCI_eps1.txt'
% load 'LLCI_eps5.txt'
% load 'LLCI_eps10.txt'
% load 'LLCI_eps25.txt'
% load 'LLCI_eps50.txt'
% load 'LLCI_eps100.txt'
% 
% subplot(2,3,1)
% plot(1:128,LLCI_eps1,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 1')
% subplot(2,3,2)
% plot(1:128,LLCI_eps5,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 5')
% subplot(2,3,3)
% plot(1:128,LLCI_eps10,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 10')
% subplot(2,3,4)
% plot(1:128,LLCI_eps25,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 25')
% subplot(2,3,5)
% plot(1:128,LLCI_eps50,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 50')
% subplot(2,3,6)
% plot(1:128,LLCI_eps100,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 100')

%% Plotting the samples for BI

% load 'BI_eps1.txt'
% load 'BI_eps5.txt'
% load 'BI_eps10.txt'
% load 'BI_eps25.txt'
% load 'BI_eps50.txt'
load 'BI_eps100.txt'
% 
% subplot(2,3,1)
% plot(1:128,BI_eps1,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 1')
% subplot(2,3,2)
% plot(1:128,BI_eps5,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 5')
% subplot(2,3,3)
% plot(1:128,BI_eps10,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 10')
% subplot(2,3,4)
% plot(1:128,BI_eps25,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 25')
% subplot(2,3,5)
% plot(1:128,BI_eps50,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 50')
% subplot(2,3,6)
% plot(1:128,BI_eps100,'+')
% ylabel('True class confidence')
% xlabel('Noise standard deviation')
% title('\epsilon = 100')

load 'BI2_eps100.txt'
load 'orig_noisy.txt'
load 'orig_noisy_RN50.txt'
plot(1:128,smooth(mean(orig_noisy,2),9))
ylabel('True class confidence')
xlabel('Noise standard deviation')
hold on
plot(1:128,smooth(mean(BI_eps100,2),9))
hold on
plot(1:128,smooth(mean(BI2_eps100,2),9)) 
ylabel('True class confidence')
xlabel('Noise standard deviation')
legend('Original + noise','Adversarial BI + noise','Original + noise (ResNet50)')

% plot(1:128,BI2_eps100,'+')