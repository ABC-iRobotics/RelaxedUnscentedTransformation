clc
clear all
%close all
RawCPUtimes_16158901705884523;
groundtruth;
Simulator_output_UT_nolandmarks_16159755757390139;
Simulator_output_newUT_nolandmarks_16159755764598398;
Simulator_output_UKF_16159755780102226
Simulator_output_newUKF_16159755829474851
Simulator_output_UT_withlandmarks_16159755848525655
Simulator_output_newUT_withlandmarks_16159755898041487
Simulator_output_AUKF_16159755916710927
Simulator_output_newAUKF_16159756020444645
%% Plot Raw CPU times
%figure,
%plot(N,old_ms/10000,'-o'), hold on, plot(N,new_ms/10000, '-o'), legend('Original UT', 'Proposed method'),grid on, ylabel('cputime [ms]'), xlabel('Num. of landmarks'); ylim([0 7]), xlim([1 51])

%% 
plotsimresult(r,path_groundtruth_t_x_y_phi,...
    path_newUT_nolandmarks_t_x_y_phi,x_newUT_nolandmarks,Sx_newUT_nolandmarks,...
    path_UT_nolandmarks_t_x_y_phi,x_UT_nolandmarks,Sx_UT_nolandmarks,false);
legend('groundtruth','original UT','proposed method')
%%
plotsimresult(r,path_groundtruth_t_x_y_phi,...
    path_newUT_withlandmarks_t_x_y_phi,x_newUT_withlandmarks,Sx_newUT_withlandmarks,...
    path_UT_withlandmarks_t_x_y_phi,x_UT_withlandmarks,Sx_UT_withlandmarks,true);
legend('groundtruth','original UT','proposed method')
%%
plotsimresult(r,path_groundtruth_t_x_y_phi,...
    path_newUKF_t_x_y_phi,x_newUKF,Sx_newUKF,...
    path_UKF_t_x_y_phi,x_UKF,Sx_UKF,true);
legend('groundtruth','original UKF','UKF based on the prop. meth.')
%%
plotsimresult(r,path_groundtruth_t_x_y_phi,...
    path_newAUKF_t_x_y_phi,x_newUKF,Sx_newAUKF,...
    path_AUKF_t_x_y_phi,x_UKF,Sx_AUKF,true);
legend('groundtruth','original AUKF','AUKF based on the prop. meth.')
%%
simtimes = [Tsim_ms_UT_nolandmarks  Tsim_ms_UT_withlandmarks Tsim_ms_UKF Tsim_ms_AUKF;...
    Tsim_ms_newUT_nolandmarks   Tsim_ms_newUT_withlandmarks Tsim_ms_newUKF Tsim_ms_newAUKF]
%%
function plotsimresult(lms,pathgt,...
    pathnew,xnew,Sxnew,...
    pathold,xold,Sxold,plotlandmarks)
    gt_xy = pathgt(1:end,2:3);
    newUT_xy = pathnew(1:end,2:3);
    oldUT_xy = pathold(1:end,2:3);

    figure,
    plot(gt_xy(:,1),gt_xy(:,2),'k'); hold on % Groundtruth trajectory
    plot(newUT_xy(:,1),newUT_xy(:,2),'r'); % newUT trajectory
    plot(oldUT_xy(:,1),oldUT_xy(:,2),'b'); % oldUT trajectory
    legend('Groundtruth', 'new UT','original UT')
    % plot endpose uncertainty
    plotxy(xnew(1:2),Sxnew(1:2,1:2),'r');
    plotxy(xold(1:2),Sxold(1:2,1:2),'b');
    if plotlandmarks
        xy_lm = zeros(length(lms),2);
        for i=1:length(lms) % plot landmark groundtruth
            xy_lm(i,:) = [lms{i}(1),lms{i}(2)];
        end
        plot(xy_lm(:,1),xy_lm(:,2),'kx')
        % plot landmark uncertainties
        for i=1:(length(xnew)-3)/2
            indices = 1+2*i+(1:2);
            plotxy(xnew(indices),Sxnew(indices,indices),'r');
        end
        for i=1:(length(xold)-3)/2
            indices = 1+2*i+(1:2);
            plotxy(xold(indices),Sxold(indices,indices),'b');
        end
    end
    grid on
    xlabel('x[m]'), ylabel('y[m]')
    set(gcf,'renderer','painters');
end

function plotxy(xy,Sxy,color)
    plot(xy(1),xy(2),[color 'x']);
    [rx,ry,phi] = cov2ell(Sxy);
    [x,y] = getsegments(xy(1),xy(2),rx,ry,phi);
    plot(x,y,[color '-']);
end

function [x,y] = getsegments(xcenter,ycenter,rx,ry,ang)
    co=cos(ang);
    si=sin(ang);
    the=linspace(0,2*pi,40+1);
    co_the = cos(the);
    si_the = sin(the);
    x=rx*co_the*co-si*ry*si_the+xcenter;
    y=rx*co_the*si+co*ry*si_the+ycenter;
end

function [rx,ry,phi] = cov2ell(S)

    if S(1,1)~=S(2,2)
        phi = atan(2*S(1,2)/(S(1,1)-S(2,2)))/2;

        x = cos(phi)^2 / (cos(phi)^2 - sin(phi)^2);
        y = 1 - x;

        rx = x*S(1,1) + y*S(2,2);
        ry = y*S(1,1) + x*S(2,2);
    else
        phi = pi/4;
        rx = (S(1,1)+S(2,2))/2 + S(1,2);
        ry = (S(1,1)+S(2,2))/2 - S(1,2);
    end
    rx = sqrt(rx);
    ry = sqrt(ry);
end