%��Ϊ��ߺ���������ص�
%% �������
clc;
P_M_Arr = 0.1:0.1:0.9;
P_F_Arr = ones(1,9) - P_M_Arr;
err_in_train = zeros(1,9);
err_in_test = zeros(1,9);
for turn = 1:9
    P_M = P_M_Arr(1,turn);  %�������������
    P_F = P_F_Arr(1,turn);  %Ů�����������
    fileID = fopen('train_data.txt');
    data = textscan(fileID,'%f\t%f\t%c');
    fclose(fileID);
    H = data{1};
    W = data{2};
    Gender = data{3};
    X = [H W];
    m = size(X,1);
    Ix = find(Gender == 'm');
    Male = X(Ix,:);
    Female = X;
    Female(Ix,:) = [];
    Male = Male'; Female = Female';
    ave_M = mean(Male')';
    ave_F = mean(Female')';
    sigma_M = cov(Male')*(length(Male)-1)/length(Male);
    sigma_F = cov(Female')*(length(Female)-1)/length(Female);
    W_M = -inv(sigma_M)/2;
    W_F = -inv(sigma_F)/2;
    w_M = inv(sigma_M)*ave_M;
    w_F = inv(sigma_F)*ave_F;
    w0_M = -1/2*ave_M'*inv(sigma_M)*ave_M - 1/2*log(det(sigma_M)) + log(P_M);
    w0_F = -1/2*ave_F'*inv(sigma_F)*ave_F - 1/2*log(det(sigma_F)) + log(P_F);
   

    %% ��ѵ�����Ͻ��в���
    gender_test = zeros(length(Gender),1);
    X = X';
    for i = 1:length(X)
        g1 = X(:,i)'*W_M*X(:,i) + w_M'*X(:,i) + w0_M;
        g2 = X(:,i)'*W_F*X(:,i) + w_F'*X(:,i) + w0_F;
        if(g1 >= g2)
            gender_test(i,1) = 1;   %��Ϊ����
        else
            gender_test(i,1) = 0;   %��ΪŮ��
        end
    end
    gender_input = zeros(length(Gender),1);
    Ix = find(Gender == 'm');
    gender_input(Ix) = 1;
    Train_Err = gender_input - gender_test;
    train_err = length(find(Train_Err ~= 0))/length(Train_Err)*100;
%     disp(['ѵ���������Ϊ',num2str(train_err),' %']);
    err_in_train(1,turn) = train_err;

    %% �ڲ��Լ��Ͻ��в���
    fileID = fopen('dataset1.txt');
    test = textscan(fileID,'%f\t%f\t%c');
    fclose(fileID);
    T = [test{1} test{2}];
    T = T';
    gender_test = zeros(length(T),1);
    for i = 1:length(T)
        g1 = T(:,i)'*W_M*T(:,i) + w_M'*T(:,i) + w0_M;
        g2 = T(:,i)'*W_F*T(:,i) + w_F'*T(:,i) + w0_F;
        if(g1 >= g2)
            gender_test(i,1) = 1;   %��Ϊ����
        else
            gender_test(i,1) = 0;   %��ΪŮ��
        end
    end
    gender_input = zeros(length(test{3}),1);
    Ix = find(test{3} == 'M');
    gender_input(Ix) = 1;
    Test_Err = gender_input - gender_test;
    test_err = length(find(Test_Err ~= 0))/length(Test_Err)*100;
%     disp(['���Լ������Ϊ',num2str(test_err),' %']);
    err_in_test(1,turn) = test_err;
    
end
