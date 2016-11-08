function MixGauss

close all;

% Load training data RGBApple, RGBNonApple, 3*I vectors
run('LoadApplesScript.m');

priorApple = 0.3;
priorNonApple = 0.7;

% Uncomment lines 12-51 (and change args in fitMixGauss's) to generate insight about different values of k from testing validation data 
% for kNonApple = 1:5
    mixGaussNonApple = fitMixGauss(RGBNonApple,4);
%     for kApple = 1:4
    %fit Gaussian model for apple data
    mixGaussApple = fitMixGauss(RGBApple,2);

    %fit Gaussian model for non-apple data
    

    % Validation
%     [imY imX imZ] = size(ValidationFig);
%     data = double(reshape(ValidationFig,imY*imX,imZ).') / 255; %3*(h*w)
%     post = posterior(data, imX, imY, mixGaussApple, mixGaussNonApple, priorApple, priorNonApple);
%     TP = 0; TN = 0; FP = 0; FN = 0;
%     for cY = 1:imY
%         for cX = 1:imX
%             if post(cY,cX) > 0.5
%                 if ValidationMask(cY,cX,2)/255 > 0.5
%                     TP = TP + 1;
%                 else
%                     FP = FP + 1;
%                 end
%             else
%                 if ValidationMask(cY,cX,2)/255 > 0.5
%                     FN = FN + 1;
%                 else
%                     TN = TN + 1;
%                 end
%             end
%         end
%     end
%     prec = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1 = 2*prec*recall/(prec+recall);
%     disp([num2str(ki), ' gaussian(s) for Apple, ', num2str(kj), ' gaussian(s) for NonApple: TP: ', num2str(TP), ', TN: ', num2str(TN), ', FP: ', num2str(FP), ', FN: ', num2str(FN), ', precision: ', num2str(prec, '%10.2e\n'), ', recall: ', num2str(recall, '%10.2e\n'), ', F1 score: ', num2str(F1, '%10.2e\n'), '.']);
%     
%     postApple = getMixGaussLogLike(RGBAppleVal,mixGaussApple)*priorApple / (getMixGaussLogLike(RGBAppleVal,mixGaussApple)*priorApple+getMixGaussLogLike(RGBAppleVal,mixGaussNonApple)*priorNonApple)
%     postNonApple = getMixGaussLogLike(RGBNonAppleVal,mixGaussNonApple)*priorNonApple / (getMixGaussLogLike(RGBNonAppleVal,mixGaussApple)*priorApple+getMixGaussLogLike(RGBNonAppleVal,mixGaussNonApple)*priorNonApple)
%     end
% end

run('LoadApplesTestScript.m');
nImages = 5;
for n = 1:nImages

    %load in test image and ground truth.
    resizefactor = 0.5;
    im = imresize(imread(Iapples{n}),resizefactor);
    hasmask = ~isempty(IapplesMasks{n});
    if hasmask
        gt = imresize(imread(IapplesMasks{n}),resizefactor);
    end

    %display test image and ground truth;
    figure; set(gcf,'Color',[1 1 1]);
    if hasmask
        subplot(1,3,1); imagesc(im); axis off; axis image;
        subplot(1,3,2); imagesc(gt); colormap(gray); axis off; axis image;
    else
        subplot(1,2,1); imagesc(im); axis off; axis image;
    end
    drawnow;

    [imY imX imZ] = size(im);
    data = double(reshape(im,imY*imX,imZ).') / 255; %3*(h*w)
    posteriorApple = posterior(data, imX, imY, mixGaussApple, mixGaussNonApple, priorApple, priorNonApple);
    if hasmask
        TP = 0; TN = 0; FP = 0; FN = 0;
        for cY = 1:imY
            for cX = 1:imX
                if posteriorApple(cY,cX) > 0.5
                    if gt(cY,cX,2)/255 > 0.5
                        TP = TP + 1;
                    else
                        FP = FP + 1;
                    end
                else
                    if gt(cY,cX,2)/255 > 0.5
                        FN = FN + 1;
                    else
                        TN = TN + 1;
                    end
                end
            end
        end
        prec = TP/(TP+FP);
        recall = TP/(TP+FN);
        F1 = 2*prec*recall/(prec+recall);
    disp(['Image #', num2str(n), ': TP: ', num2str(TP), ', TN: ', num2str(TN), ', FP: ', num2str(FP), ', FN: ', num2str(FN), ', precision: ', num2str(prec, '%10.2e\n'), ', recall: ', num2str(recall, '%10.2e\n'), ', F1 score: ', num2str(F1, '%10.2e\n'), '.']);
    end
    
    %draw posterior
    clims = [0, 1];
    subplot(1,3,3); imagesc(posteriorApple, clims); colormap(gray); axis off; axis image; drawnow;
    % set(gca, 'clim', [0, 1]);
end

%print values
mixGaussApple.weight
mixGaussApple.mean
mixGaussApple.cov
mixGaussNonApple.weight
mixGaussNonApple.mean
mixGaussNonApple.cov

%==========================================================================
%==========================================================================

function like = calcGaussianProb(data, mixGauss)
like = 0;
D = length(data);
for cGauss = 1:mixGauss.k
    like = like + mixGauss.weight(cGauss) * (1/((2*pi)^(D/2)*(norm(mixGauss.cov(cGauss)))^(1/2)))*exp(-0.5*((data-mixGauss.mean(cGauss)).')*inv(mixGauss.cov(cGauss))*(data-mixGauss.mean(cGauss)));
end

function logLike = getMixGaussLogLike(data,mixGaussEst);

%find total number of data items
nData = size(data,2);
nDim = size(data,1);

%initialize log likelihoods
logLike = 0;

%run through each data item
for (cData = 1:nData)
    thisData = data(:,cData);
    
    like = 0;
    for(cGauss = 1:mixGaussEst.k)
        like = like + mixGaussEst.weight(cGauss) * 1/((2*pi)^(nDim/2)*(norm(mixGaussEst.cov(:,:,cGauss)))^(1/2)) * exp(-0.5*((thisData-mixGaussEst.mean(cGauss)).')*inv(mixGaussEst.cov(:,:,cGauss))*(thisData-mixGaussEst.mean(cGauss)));
    end
    
    %add to total log like
    logLike = logLike+log(like);        
end;

function posteriorApple = posterior(data, imX, imY, mixGaussApple, mixGaussNonApple, priorApple, priorNonApple)
%determine posterior for each pixel being apple
posteriorApple = zeros(imY,imX);
for (cY = 1:imY); 
    % fprintf('Processing Row %d\n',cY);     
    for (cX = 1:imX);          
        %extract this pixel data
        thisPixelData = data(:,(cX-1)*imY+cY);
        %calculate likelihood of this data given apple model
        likeApple = calcGaussianProb(thisPixelData,mixGaussApple);
        %calculate likelihood of this data given non-apple model
        likeNonApple = calcGaussianProb(thisPixelData,mixGaussNonApple);
        %calculate posterior probability
        num = likeApple*priorApple;
        den = num + likeNonApple*priorNonApple;
        posteriorApple(cY,cX) = num/den;
    end;
end;

function mixGaussEst = fitMixGauss(data,k);
        
[nDim nData] = size(data);

%MAIN E-M ROUTINE 
%there are nData data points, and there is a hidden variable associated
%with each.  If the hidden variable is 0 this indicates that the data was
%generated by the first Gaussian.  If the hidden variable is 1 then this
%indicates that the hidden variable was generated by the second Gaussian
%etc.

postHidden = zeros(k, nData);

%in the E-M algorithm, we calculate a complete posterior distribution over
%the (nData) hidden variables in the E-Step.  In the M-Step, we
%update the parameters of the Gaussians (mean, cov, w).  

%we will initialize the values to random values
mixGaussEst.d = nDim;
mixGaussEst.k = k;
mixGaussEst.weight = (1/k)*ones(1,k);
mixGaussEst.mean = 2*randn(nDim,k);
for (cGauss =1:k)
    mixGaussEst.cov(:,:,cGauss) = (0.5+1.5*rand(1))*eye(nDim,nDim);
end;

%calculate current likelihood
logLike = getMixGaussLogLike(data,mixGaussEst);
    fprintf('Log Likelihood Iter 0 : %4.3f\n',logLike);

nIter = 10;
for (cIter = 1:nIter)
    broken = 0;
    %Expectation step
   
   for (cData = 1:nData)
        %TO DO (g): fill in column of 'hidden' - calculate posterior probability that
        %this data point came from each of the Gaussians1
        %replace this:
        ll = zeros(1,k);
        thisdata = data(:,cData);
        for cGauss=1:k
            ll(cGauss) = mixGaussEst.weight(cGauss) * 1/((2*pi)^(mixGaussEst.d/2)*(norm(mixGaussEst.cov(:,:,cGauss)))^(1/2))*exp(-0.5*((thisdata-mixGaussEst.mean(:,cGauss)).')*inv(mixGaussEst.cov(:,:,cGauss))*(thisdata-mixGaussEst.mean(:,cGauss)));
        end
        
        for cGauss=1:k
            postHidden(cGauss,cData) = ll(cGauss) ./ sum(ll);
        end
   end;
   
   %Maximization Step
   
   %for each constituent Gaussian
   for (cGauss = 1:k) 
        %TO DO (h):  Update weighting parameters mixGauss.weight based on the total
        %posterior probability associated with each Gaussian. Replace this:
        sumIr = sum(postHidden(cGauss,:));
        
        mixGaussEst.weight(cGauss) = sumIr/sum(sum(postHidden(:,:)));
        if mixGaussEst.weight(cGauss) < 0.01
            % || mixGaussEst.weight(cGauss) > 0.8
            broken = 1;
            break
        end
        
        %TO DO (i):  Update mean parameters mixGauss.mean by weighted average
        %where weights are given by posterior probability associated with
        %Gaussian.  Replace this:
        ji = data(:,:) .* postHidden(cGauss,:);
        mixGaussEst.mean(:,cGauss) = sum(ji,2) ./ sumIr;
        
        %TO DO (j):  Update covariance parameter based on weighted average of
        %square distance from update mean, where weights are given by
        %posterior probability associated with Gaussian
        jj = data - mixGaussEst.mean(:,cGauss);
        jk = zeros(nDim,nDim,nData);
        for i=1:nData
            jk(:,:,i) = jj(:,i) * jj(:,i).' .* postHidden(cGauss,i);
        end
        mixGaussEst.cov(:,:,cGauss) = sum(jk,3) ./ sumIr;
   end;
   if (broken == 1)
       break;
   end

   %calculate the log likelihood
   logLike = getMixGaussLogLike(data,mixGaussEst);
   fprintf('Log Likelihood Iter %d : %4.3f\n',cIter,logLike);

end;
if (broken == 1)
    mixGaussEst = fitMixGauss(data,k);
end