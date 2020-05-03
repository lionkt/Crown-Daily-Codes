test_vehicle_ratio = 0.8:0.2:2;
acc_speed_rmse = [1.506,1.46,1.45,1.45,1.451,1.453]; %0.8����ײ
acc_dist_error = [0.81,0.758,0.76,0.76,0.76,0.759]; %0.8����ײ
cacc_speed_rmse = [0.69,0.604,0.59,0.59,0.59,0.594,0.596]; 
cacc_dist_error = [1.079,0.349,0.331,0.334,0.333,0.333,0.333];
cacc_into_bound_time = [64.8, 64.35, 65, 64.9, 64.9,65]; %0.8û��������
rl_speed_rmse = [0.594,0.37,0.288,0.24,0.207,0.182,0.163];
rl_dist_error=[0.34,0.266,0.232,0.211,0.198,0.188,0.18];
rl_into_bound_time=[63.95,63.45,63.2,62.4,62.5,61.9,62.4];

%% plot
figure();
plot(test_vehicle_ratio(2:end), acc_speed_rmse,'--*','linewidth',1.5); hold on;
plot(test_vehicle_ratio, cacc_speed_rmse, '--+','linewidth',1.5);
plot(test_vehicle_ratio, rl_speed_rmse, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
ylim([0.1, 1.6]);
legend('CTG-ACC','��ģCACC','����������DDPG');
grid on;
xlabel('����p_{veh}ȡֵ');
ylabel('���۳����캽���ٶ�RMSE (m/s)');

%% plot 
figure();
plot(test_vehicle_ratio(2:end), acc_dist_error,'--*','linewidth',1.5); hold on;
plot(test_vehicle_ratio, cacc_dist_error, '--+','linewidth',1.5);
plot(test_vehicle_ratio, rl_dist_error, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
ylim([0.1, 1.2]);
legend('CTG-ACC','��ģCACC','����������DDPG');
grid on;
xlabel('����p_{veh}ȡֵ');
ylabel('���Ӽ�����\epsilon (m)');

%% plot
figure();
plot(test_vehicle_ratio(2:end), cacc_into_bound_time, '--+','linewidth',1.5); hold on;
plot(test_vehicle_ratio, rl_into_bound_time, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
% ylim([0.1, 1.2]);
legend('��ģCACC','����������DDPG');
grid on;
xlabel('����p_{veh}ȡֵ');
ylabel('���ӽ���5%����ƽ����ʱ (s)');
