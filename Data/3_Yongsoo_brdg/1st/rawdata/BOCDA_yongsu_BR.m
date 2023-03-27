
clc, clear, close all
screensize = get(0,'screensize') ;
fig01 = figure ; set(fig01,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1.2 screensize(4)/1.4])
fig02 = figure ; set(fig02,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1 screensize(4)/1])
fig03 = figure ; set(fig03,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1 screensize(4)/1])
fig04 = figure ; set(fig04,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1 screensize(4)/0.5])
fig05 = figure ; set(fig05,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/2 screensize(4)/1.5])
bit_x = [ 89 645 ; 808 1364]  ;
bit_y = [10.85 10.95]  ;
% color = [78 205 196 ; 255 107 107 ;85 98 112 ;199 244 100;199 77 88]./256;
Marker = {'o','^','s','*'} ;
nofT = 1:17;

for j = 1:length(nofT)
    str03 = ['LC00_',num2str(nofT(j)),'.txt'] ;
    tmp00  = importdata(str03);
    disp(str03)
    bit = transpose(tmp00.data(1,:)) ;
    tmp00 = tmp00.data(2,:) ;
    tmp01(:,j) = transpose(tmp00) ;
    %     result.ref(:,j) = tmp00 ;
end

for i =1:size(tmp01,1)
    result.ref(i,:) = filloutliers(tmp01(i,:),'linear');
end
figure(fig01)
hold on ; grid on ;
plot(result.ref)
xlabel('bit','FontSize',24,'FontWeight','Bold')
ylabel('Brillouin Freq.','FontSize',24,'FontWeight','Bold')
set(gca,'FontSize',24,'LineWidth',2,'FontName','Times New Roman')
xlabel('bit','FontSize',24,'FontWeight','Bold')
ylabel('Brillouin Freq.','FontSize',24,'FontWeight','Bold')
set(gca,'FontSize',24,'LineWidth',2)
% lgd = legend ;
% lgd.Location = 'NorthOutside' ;
% lgd.Orientation = 'Horizontal' ;
% lgd.NumColumns = 6 ;
% formatSpec = '%3.3f';


LC = [7 1 2 3 5  ] ;
for i = 1:length(LC)
    if LC(i) == 5
        nofT = 1:5 ;
        for j = 1:length(nofT)
            str03 = ['LC0',num2str(LC(i)),'_',num2str(nofT(j)),'.txt'] ;
            tmp00  = importdata(str03);
            disp(str03)
            bit = transpose(tmp00.data(1,:)) ;
            tmp00 = tmp00.data(2,:) ;
            tmp02(:,j) = transpose(tmp00) ;
        end
        for k =1:size(tmp02,1)
            result.sig{i}(k,:) = filloutliers(tmp02(k,:),'linear');
        end
    else
        nofT = 1:3 ;
        for j = 1:length(nofT)
            str03 = ['LC0',num2str(LC(i)),'_',num2str(nofT(j)),'.txt'] ;
            tmp00  = importdata(str03);
            disp(str03)
            bit = transpose(tmp00.data(1,:)) ;
            tmp00 = tmp00.data(2,:) ;
            tmp02(:,j) = transpose(tmp00) ;
        end
        for k =1:size(tmp02,1)
            result.sig{i}(k,:) = filloutliers(tmp02(k,:),'linear');
        end
    end
    clear tmp02
end

ref_p = [1 3 ; 4 6 ; 7 9 ; 10 12 ; 13 17 ] ;
for i = 1:length(LC)
    for j = 1:size(bit_x,2)
        if j == 1
            result.strain{i}(:,j) =  mean(result.sig{i}(bit_x(j,1):bit_x(j,2),:),2) ...
                - mean(result.ref(bit_x(j,1):bit_x(j,2),ref_p(i,1):ref_p(i,2)),2);
        else
            result.strain{i}(:,j) = mean(result.sig{i}(bit_x(j,2):-1:bit_x(j,1),:),2) ...
                - mean(result.ref(bit_x(j,2):-1:bit_x(j,1),ref_p(i,1):ref_p(i,2)),2) ;
        end
    end
end


for i = 1:length(LC)
    %             figure(fig01)
    %             patch([bit_x(j,1) bit_x(j,2) bit_x(j,2) bit_x(j,1)],[bit_y(1) bit_y(1) bit_y(2) bit_y(2)],'y','FaceAlpha',.1,'LineWidth',2,'DisplayName',['rod#0',num2str(j)])
    for j = 1:size(bit_x,1)
        figure(fig02)
        %         if j ==1
        %             subplot(5,2,2*i-1)
        %         else
        %             subplot(5,2,2*i)
        %         end
        subplot(5,1,i)
        plot(result.strain{i}(:,j)*2e4,'-','LineWidth',2,'MarkerSize',10)
        grid on ; hold on ;axis tight
        %         plot(movmean(result.strain{i}(:,j),10)*2e4,'--o','LineWidth',2,'MarkerSize',10)
        ax01 = gca ;
        ax01.YLabel.String = 'Strain(\mu\epsilon)' ;
        ax01.YLabel.FontSize = 32 ;
        ax01.YLabel.FontWeight = 'bold' ;
        ax01.FontSize = 32 ;
        ax01.FontWeight = 'bold' ;
        ax01.LineWidth = 3 ;
        ax01.YLim = [-30 50] ;
        ax01.FontName = 'Times New Roman' ;
    end
end
%%
% figure(fig03)
% for i =1:length(nofT)-1
%     for j =1:5
%         subplot(4,4,i)
%         grid on ; hold on ;axis tight
%         if rem(j,2) == 0
%             tmp02{i}(:,j) = flip(result.strain(bit_x(j,1):bit_x(j,2),i))*2e4 ;
%         else
%             tmp02{i}(:,j) = result.strain(bit_x(j,1):bit_x(j,2),i)*2e4 ;
%         end
%         plot(tmp02{i}(:,j),'-o','LineWidth',2,'MarkerSize',10,'DisplayName',['rod#0',num2str(j)])
%         ax01 = gca ;
%         ax01.YLabel.String = 'Strain(\mu\epsilon)' ;
%         ax01.YLabel.FontSize = 32 ;
%         ax01.YLabel.FontWeight = 'bold' ;
%         ax01.FontSize = 32 ;
%         ax01.FontWeight = 'bold' ;
%         ax01.LineWidth = 3 ;
%         ax01.Title.String = [num2str(nofT(i+1)),'KN'] ;
%         ax01.FontSize = 32 ;
%         ax01.FontWeight = 'bold' ;
%         ax01.YLim = [-inf Inf] ;
%         ax01.FontName = 'Times New Roman' ;
%     end
% end

grid_x = 300:300:600;
grid_y = 1100:50:556*50+1100 ;
[X,Y] =meshgrid(grid_x,grid_y) ;
figure(fig04)
tiledlayout(2,3)
filename = 'surf_slab.gif' ;
for j =1:5
    figure(fig04)
    nexttile
    tmp03 = movmean(result.strain{j},10)*2e4 ;
    TF = find(tmp03> 50) ;
    tmp03(TF) = NaN ;
    surfc(X,Y,tmp03)
    shading interp ;
    colormap jet ;
    axis tight ;
    caxis([-20 40]);
    a = colorbar;
    a.Location='southoutside';
    a.Label.String ='Strain(\mu\epsilon)' ;
    %     a.Limits = [0 4000] ;
    ax01 = gca ;
    ax01.XLabel.String = 'x(mm)' ;
    ax01.XLabel.FontSize = 24 ;
    ax01.YLabel.String = 'y(m)' ;
    ax01.YLabel.FontSize = 24 ;
    ax01.YLabel.FontWeight = 'bold' ;
    ax01.FontSize = 24 ;
    ax01.FontWeight = 'bold' ;
    ax01.LineWidth = 3 ;
    ax01.Title.String = ['LC-',num2str(LC(j))];
    ax01.Title.FontWeight = 'bold' ;
    ax01.FontSize = 24 ;
    ax01.FontWeight = 'bold' ;
    ax01.YTick = 0:5000:27750;
    ax01.YTickLabel = 0:5:27.75;
    ax01.XTick = grid_x ;
    ax01.XTickLabel = [1.92 5.76];
    ax01.FontName = 'Times New Roman' ;
    %     axis equal
    ax01.XLim = [250 650];
    ax01.YLim = [0 30000];
    view(-55,90)
    
    
    %     figure(fig05)
    %     surfc(X,Y,tmp02{j})
    %     shading interp ;
    %     colormap jet ;
    %     axis tight ;
    %     caxis([0 3500]);
    %     %     a = colorbar;
    %     %     a.Location='southoutside';
    %     %     a.Label.String ='Strain(\mu\epsilon)' ;
    %     ax02 = gca ;
    %     ax02.XLabel.String = 'x(mm)' ;
    %     ax02.XLabel.FontSize = 24 ;
    %     ax02.YLabel.String = 'y(mm)' ;
    
    %     ax02.YLabel.FontSize = 24 ;
    %     ax02.YLabel.FontWeight = 'bold' ;
    %     ax02.FontSize = 24 ;
    %     ax02.FontWeight = 'bold' ;
    %     ax02.LineWidth = 3 ;
    %     ax02.Title.String = [num2str(nofT(j+1)),'KN'] ;
    %     ax02.Title.FontWeight = 'bold' ;
    %     ax02.FontSize = 24 ;
    %     ax02.FontWeight = 'bold' ;
    %     ax02.YTick = 0:400:2400;
    %     ax02.XTick = grid_x ;
    %     ax02.FontName = 'Times New Roman' ;
    %     axis equal
    %     ax02.XLim = [300 2700];
    %     view(-55,90)
    %
    %     drawnow
    %     frame= getframe(fig05) ;
    %     im = frame2im(frame) ;
    %     [imind,cm] = rgb2ind(im,256) ;
    %     if j==1
    %         imwrite(imind,cm,filename,'gif','Loopcount',inf)
    %     else
    %         imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',1)
    %     end
    %
end
saveas(fig02,'Rod','png')
saveas(fig01,'Freq','png')
saveas(fig04,'contour','png')

%%
FBG = load('LC_FBG.mat') ;
% BOCDA_L = 1.1:0.05:30-1.1 ;
clf
FBG_L = 2400:1800:30000-2400 ;
LC_case = [7 1:3 5 ] ;
for i =1:5
    subplot(5,1,i)
    hold on
    for j = 1:3
        for k = 1:2
            plot(grid_y,movmean(result.strain{i}(:,k),20)*2e4,'s','MarkerSize',6,'MarkerFace','k','MarkerEdgeColor','k')
        end
        plot(FBG_L,FBG.result(i).str{j}(1:15),'-ob','LineWidth',2,'MarkerSize',12,'MarkerFace','b','MarkerEdgeColor','k')
        plot(FBG_L,FBG.result(i).str{j}(16:30),'-sr','LineWidth',2,'MarkerSize',12,'MarkerFace','r','MarkerEdgeColor','k')
        ax01 = gca ;
        grid on
        if i ==3
            ax01.YLabel.String = 'Strain(\mu\epsilon)' ;
            ax01.YLabel.FontSize = 24 ;
            ax01.YLim = [-inf 45] ;
        elseif i == 5
            ax01.YLim = [-inf 40] ;
        else
            ax01.YLabel.String = [];
            ax01.YLim = [-inf 30] ;
        end
        ax01.YLabel.FontWeight = 'bold' ;
        ax01.FontSize = 24 ;
        ax01.FontWeight = 'bold' ;
        ax01.LineWidth = 3 ;
%         ax01.Title.String = ['LC#0',num2str(LC_case(i))] ;
        ax01.FontSize = 24 ;
        ax01.FontWeight = 'bold' ;
        ax01.FontName = 'Times New Roman' ;
    end
end