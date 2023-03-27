clear ; close all ; clc
screensize = get(0,'screensize') ;
fig01 = figure ; set(fig01,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1.2 screensize(4)/1.4])
fig02 = figure ; set(fig02,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1 screensize(4)/1])
fn = { 'LC-1st.20221114231815.txt' , ...
    'LC-2nd.20221114235406.txt' , ...
    'LC-3rd.20221115002211.txt', ...
    'LC-4th.20221115004819.txt',...
    'LC-5th.20221115011651.txt'} ;


for k=1:length(fn)
    T = readtable(fn{k}) ;
    data = table2array(T(:,6:end-1)) ;
    figure(fig01)
    plot(data(:,7)) ;
    if k == 5
        nTest = 5;
    else
        nTest = 3 ;
    end
    for j = 1:nTest
        [x,~] = ginput(2) ;
        pos_x = round(x) ;
        [x,~] = ginput(2) ;
        pos_x01 = round(x) ;
        for i = 1:size(data,2)
            ref = mean(data(pos_x(1):pos_x(2),i)) ;
            result(k).str{j}(:,i) = (-1/0.78*(1-mean(data(pos_x01(1):pos_x01(2),i))./ref))*10^6 ;
        end
    end
    figure(fig02)
    subplot(5,1,k)
    plot(result(k).str{j}(1:15),':ob')
    hold on
    plot(result(k).str{j}(16:30),'-rs')
end

%%
screensize = get(0,'screensize') ;
% fig01 = figure ; set(fig01,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1.2 screensize(4)/1.4])
fig02 = figure ; set(fig02,'position',[screensize(3)/10 screensize(4)/10 screensize(3)/1 screensize(4)/1])
load('LC.mat')
LC_case =  [7 1:3 5]  ;
for i = 1:5
    for j =1:size(result(i).str,2)
        figure(fig02)
        subplot(5,1,i)
        plot(result(i).str{j}(1:15),'--ob','LineWidth',2,'MarkerSize',12,'DisplayName','Left','MarkerFace','b','MarkerEdgeColor','k')
        hold on
        plot(result(i).str{j}(16:30),'-sr','LineWidth',2,'MarkerSize',12,'DisplayName','Right','MarkerFace','r','MarkerEdgeColor','k')
        grid on ; axis tight
    end
    ax01 = gca ;
    if i ==3 
        ax01.YLabel.String = 'Strain(\mu\epsilon)' ;
        ax01.YLabel.FontSize = 24 ;
    elseif i ==1
        legend({'Left','Right'}) 
%         lgd.String = lgd.String{1:2}  ;
%         lgd.Orientation = 'Horizontal' ; 
    else
        ax01.YLabel.String = [];
    end
    ax01.YLabel.FontWeight = 'bold' ;
    ax01.FontSize = 24 ;
    ax01.FontWeight = 'bold' ;
    ax01.LineWidth = 3 ;
    ax01.Title.String = ['LC#0',num2str(LC_case(i))] ;
    ax01.FontSize = 24 ;
    ax01.FontWeight = 'bold' ;
    ax01.YLim = [-inf 35] ;
    ax01.FontName = 'Times New Roman' ;
end