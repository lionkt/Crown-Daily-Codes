clc;
clear;
path = '../OutputImg/';
speed_path = [path,'speed_data.txt'];
acc_path = [path, 'acc_data.txt'];
jerk_path = [path, 'jerk_data.txt'];
location_path = [path, 'location_data.txt'];
inter_dist_path = [path, 'inter-distance_data.txt'];
speed_data = importdata(speed_path);
acc_data = importdata(acc_path);
jerk_data = importdata(jerk_path);
location_data = importdata(location_path);
inter_dist_data = importdata(inter_dist_path);
%% ��·���
Time_Step = 0.2;
Car_Length = 5;
Desired_inter_distance = 5;
MAX_V = 60/3.6; %m/s
ROAD_LENGTH = MAX_V * 70;
START_LEADER_TEST_DISTANCE = ROAD_LENGTH / 2;


%% plot data

disp('��ʼ���Ժ�����');
disp('======location���Ϊ:');
sumup =  0;
ixx = find(location_data(:,1)>=START_LEADER_TEST_DISTANCE);
ixx1 = ixx;
if max(ixx1)>size(inter_dist_data,1)
    ixx1(ixx1>size(inter_dist_data,1)) = [];
end
for i=1:size(inter_dist_data,2);
    temp_err = inter_dist_data(ixx1,i) - ones(length(inter_dist_data(ixx1,i)),1)*(Desired_inter_distance);
    temp_rmse = sqrt(mean(temp_err.^2));
    if i>1
        sumup = sumup+temp_rmse;
    end
    disp(['��',num2str(i),'F����ǰ����location RSMEΪ��',num2str(temp_rmse),'m']);
end
disp(['�ӵ�2��F������',num2str(size(inter_dist_data,2)),'��location RMSE�ľ�ֵΪ:',num2str(sumup/(size(inter_dist_data,2)-1))]);

disp('======�ٶ����Ϊ:');
sumup =  0;
ixx2 = ixx;
if max(ixx2)>size(speed_data,1)
    ixx2(ixx2>size(speed_data,1)) = [];
end
leader_test_speed = speed_data(ixx2,1);
for i=2:size(location_data,2);
    temp_err = speed_data(ixx2,i)-leader_test_speed;
    temp_rmse = sqrt(mean(temp_err.^2));
    if i>1
        sumup = sumup+temp_rmse;
    end
    disp(['��',num2str(i),'F����ǰ����speed RSMEΪ��',num2str(temp_rmse),'m/s']);
end
disp(['�ӵ�2��F������',num2str(size(inter_dist_data,2)),'��speed RMSE�ľ�ֵΪ:',num2str(sumup/(size(inter_dist_data,2)-1))]);


figure;
% suptitle('dynamics');
subplot(211);
time1 = [1:1:size(speed_data,1)]*Time_Step;
max_speed_arr = ones(size(speed_data,1),1)*MAX_V;
plot(time1,max_speed_arr,':r','linewidth',1.7);
hold on;
for i=1:size(speed_data,2)
    plot(time1, speed_data(:,i),'linewidth',1.7);
    hold on;
end
title('velocity');
ylabel('m/s');
% set(gca,'xticklabel',[]);    %����x��
grid on;

subplot(212);
time2 = [1:1:size(acc_data,1)]*Time_Step;
for i=1:size(acc_data,2)
    plot(time1, acc_data(:,i),'linewidth',1.7);
    hold on;
end
title('acceleration');
xlabel('time stam(s)');ylabel('m/s^2');
grid on;
% subplot(313);
% time3 = [1:1:size(jerk_data,1)]*Time_Step;
% for i=1:size(jerk_data,2)
%     plot(time3, jerk_data(:,i),'linewidth',1.7);
%     hold on;
% end
% grid on;


figure;
% suptitle('location');
subplot(211);
time1 = [1:1:size(inter_dist_data,1)]*Time_Step;
des_inter_dist = ones(size(inter_dist_data,1),1)*Desired_inter_distance;
plot(time1,des_inter_dist,':r','linewidth',1.7);
hold on;
for i=1:size(inter_dist_data,2)
    plot(time1, inter_dist_data(:,i),'linewidth',1.7);
    hold on;
end
title('Inter-vehicle Spacing');
ylabel('m');xlabel('time stamp (s)');
grid on;
subplot(212);
for i=1:size(inter_dist_data,2)-1
    inter_dist_1 = inter_dist_data(:,i);
    inter_dist_2 = inter_dist_data(:,i+1);
    yf_norm = abs(fft(inter_dist_2) ./ fft(inter_dist_1));
    yf_norm_half = yf_norm(1:int32(length(yf_norm)/2));
    plot(yf_norm_half,'linewidth',1.7);
    hold on;
end
grid on;
title('Frequency Domain Error Ratio');
ylabel('amplitude');xlabel('frequency');



