function [ x , status ] =  SGM( f , varargin )

% function [ x , status ] = SGM( f , x , eps , astart , tau , MaxFeval ,
%                                MInf , mina )
%
% Apply the classical Subgradient Method for the minimization of the
% provided function f, which must have the following interface:
%
%   [ v , g ] = f( x )
%
% Input:
%
% - x is either a [ n x 1 ] real (column) vector denoting the input of
%   f(), or [] (empty).
%
% Output:
%
% - v (real, scalar): if x == [] this is the best known lower bound on
%   the unconstrained global optimum of f(); it can be -Inf if either f()
%   is not bounded below, or no such information is available. If x ~= []
%   then v = f(x).
%
% - g (real, [ n x 1 ] real vector): this also depends on x. if x == []
%   this is the standard starting point from which the algorithm should
%   start, otherwise it is a subgradient of f() at x (possibly the
%   gradient, but you should not apply this algorithm to a differentiable
%   f)
%
% The other [optional] input parameters are:
%
% - x (either [ n x 1 ] real vector or [], default []): starting point.
%   If x == [], the default starting point provided by f() is used.
%
% - eps (real scalar, optional, default value 1e-6): the accuracy in the 
%   stopping criterion. If eps > 0, then a target-level Polyak stepsize
%   with nonvanishing threshold is used, and eps is taken as the minimum
%   relative value for the displacement, i.e.,
%
%       delta^i >= eps * max( abs( f( x^i ) ) , 1 )
%
%   is used as the minimum value for the displacement. If eps < 0 and
%   v_* = f( [] ) > -Inf, then the algorithm "cheats" and it does an
%   exact Polyak stepsize with termination criteria
%
%       ( f^i_{ref} - v_* ) <= ( - eps ) * max( abs( v_* ) , 1 )
%
%   Finally, if eps == 0 the algorithm rather uses a DSS (diminishing
%   square-summable) stepsize, i.e., astart * ( 1 / i ) [see below]
%
% - astart (real scalar, optional, default value 1e-4): if eps > 0, i.e.,
%   a target-level Polyak stepsize with nonvanishing threshold is used,
%   then astart is used as the relative value to which the displacement is
%   reset each time f( x^{i + 1} ) <= f^i_{ref} - delta^i, i.e.,
%
%     delta^{i + 1} = astart * max( abs( f^{i + 1}_{ref} ) , 1 )
%
%   If eps == 0, i.e. a diminishing square-summable) stepsize is used, then
%   astart is used as the fixed scaling factor for the stepsize sequence
%   astart * ( 1 / i ).
%
% - tau (real scalar, optional, default value 0.95): if eps > 0, i.e.,
%   a target-level Polyak stepsize with nonvanishing threshold is used,
%   then delta^{i + 1} = delta^i * tau each time
%      f( x^{i + 1} ) > f^i_{ref} - delta^i
%
% - MaxFeval (integer scalar, optional, default value 300): the maximum
%   number of function evaluations (hence, iterations, since there is
%   exactly one function evaluation per iteration).
%
% - MInf (real scalar, optional, default value -Inf): if the algorithm
%   determines a value for f() <= MInf this is taken as an indication that
%   the problem is unbounded below and computation is stopped
%   (a "finite -Inf").
%
% - mina (real scalar, optional, default value 1e-16): if the algorithm
%   determines a stepsize value <= mina, this is taken as the fact that the
%   algorithm has already obtained the most it can and computation is
%   stopped. It is legal to take mina = 0.
%
% Output:
%
% - x ([ n x 1 ] real column vector): the best solution found so far.
%
% - status (string): a string describing the status of the algorithm at
%   termination 
%
%   = 'optimal': the algorithm terminated having proven that x is a(n
%     approximately) optimal solution; this only happens when "cheating",
%     i.e., explicitly uses v_* = f( [] ) > -Inf, unless in the very
%     unlikely case that f() spontaneously produces an almost-null
%     subgradient
%
%   = 'unbounded': the algorithm has determined an extrenely large negative
%     value for f() that is taken as an indication that the problem is
%     unbounded below (a "finite -Inf", see MInf above)
%
%   = 'stopped': the algorithm terminated having exhausted the maximum
%     number of iterations: x is the bast solution found so far, but not
%     necessarily the optimal one
%
%{
 =======================================
 Author: Antonio Frangioni
 Date: 17-11-22
 Version 1.11
 Copyright Antonio Frangioni
 =======================================
%}

Plotf = 1;
% 0 = nothing is plotted
% 1 = the level sets of f and the trajectory are plotted (when n = 2)
% 2 = the function value / gap are plotted

Interactive = false;  % if we pause at every iteration

% reading and checking input- - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if ~ isa( f , 'function_handle' )
   error( 'f not a function' );
end

if isempty( varargin ) || isempty( varargin{ 1 } )
   [ fStar , x ] = f( [] );
else
   x = varargin{ 1 };
   if ~ isreal( x )
      error( 'x not a real vector' );
   end

   if size( x , 2 ) ~= 1
      error( 'x is not a (column) vector' );
   end

   fStar = f( [] );
end

n = size( x , 1 );

if length( varargin ) > 1
   eps = varargin{ 2 };
   if ~ isreal( eps ) || ~ isscalar( eps )
      error( 'eps is not a real scalar' );
   end
else
   eps = 1e-6;
end

if eps < 0 && fStar == - Inf
   % no way of cheating since the true optimal value is unknonw
   eps = - eps;  % revert to ordinary target level stepsize
end

if length( varargin ) > 2
   astart = varargin{ 3 };
   if ~ isscalar( astart )
      error( 'astart is not a real scalar' );
   end
   if astart < 0
      error( 'astart must be > 0' );
   end       
else
   astart = 1e-4;
end

if length( varargin ) > 3
   tau = varargin{ 4 };
   if ~ isscalar( tau )
      error( 'tau is not a real scalar' );
   end
   if tau <= 0 || tau >= 1
      error( 'tau is not in (0 ,1)' );
   end       
else
   tau = 0.95;
end

if length( varargin ) > 4
   MaxFeval = round( varargin{ 5 } );
   if ~ isscalar( MaxFeval )
      error( 'MaxFeval is not an integer scalar' );
   end
else
   MaxFeval = 300;
end

if length( varargin ) > 5
   MInf = varargin{ 6 };
   if ~ isscalar( MInf )
      error( 'MInf is not a real scalar' );
   end
else
   MInf = - Inf;
end

if length( varargin ) > 6
   mina = varargin{ 7 };
   if ~ isscalar( mina )
      error( 'mina is not a real scalar' );
   end
   if mina < 0
      error( 'mina is < 0' );
   end       
else
   mina = 1e-16;
end

% initializations - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fprintf( 'Subradient method\n');
if fStar > - Inf
   fprintf( 'iter\trel gap\t\tbest gap\t|| g(x) ||\ta\n\n');
else
   fprintf( 'iter\tf(x)\t\tf best\t\t|| g(x) ||\ta\n\n');
end


if Plotf == 2
   gap = [];
   xlim( [ 0 MaxFeval ] );
   ax = gca;
   ax.FontSize = 16;
   ax.Position = [ 0.03 0.07 0.95 0.92 ];
   ax.Toolbar.Visible = 'off';
end

% main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

iter = 1;
xref = x;
fref = Inf;    % best f-value found so far
if eps > 0
   delta = 0;  % required displacement from fref;
end

while true

    % compute function and subgradient- - - - - - - - - - - - - - - - - - - - -

    [ v , g ] = f( x );
    ng = norm( g );

    if eps > 0  % target-level stepsize
       if v <= fref - delta  % found a "significantly" better point
          delta = astart * max( [ abs( v ) , 1 ] );  % reset delta
       else  % decrease delta
          delta = max( [ delta * tau , ...
                        eps * max( [ abs( min( [ v fref ] ) ) , 1 ] ) ] );
       end
    end

    if v < fref    % found a better f-value (however slightly better)
       fref = v;   % update fref
       xref = x;   % this is the incumbent solution
    end

    % output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -
  
    if fStar > - Inf
       gapk = ( v - fStar ) / max( [ abs( fStar ) 1 ] );
       bstgapk = ( fref - fStar ) / max( [ abs( fStar ) 1 ] );

       fprintf( '%4d\t%1.4e\t%1.4e\t%1.4e' , iter , gapk , bstgapk , ng );

       if Plotf == 2
          gap( end + 1 ) = gapk;
          semilogy( gap , 'Color' , 'k' , 'LineWidth' , 2 );
          ylim( [ 1e-15 1e+1 ] );
          drawnow;
       end
    else
       fprintf( '%4d\t%1.8e\t%1.8e\t\t%1.4e' , iter , fref , v , ng );

       if Plotf == 2
          gap( end + 1 ) = v;
          plot( gap , 'Color' , 'k' , 'LineWidth' , 2 );
          drawnow;
       end    
    end

    % stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -

    if eps < 0 && fref - fStar <= - eps * max( [ abs( fStar ) , 1 ] )
       xref = x;
       status = 'optimal';
       fprintf( '\n' );
       break;
    end

    if ng < 1e-12  % unlikely, but it could happen
       xref = x;
       status = 'optimal';
       fprintf( '\n' );
       break;
    end
    
    if iter > MaxFeval
       status = 'stopped';
       fprintf( '\n' );
       break;
    end

    % compute stepsize- - - - - - - - - - - - - - - - - - - - - - - - - - -

    if eps > 0        % Polyak stepsize with target level
       a = ( v - fref + delta ) / ( ng * ng );
    elseif eps < 0    % true Polyak stepsize (cheating)
       a = ( v - fStar ) / ( ng * ng );
    else              % diminishing square-summable stepsize
       a = astart * ( 1 / iter );
    end
        
    % output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

    fprintf( '\t%1.4e' , a );
    fprintf( '\n' );

    if a <= mina
       status = 'stopped';
       fprintf( '\n' );
       break;
    end
    
    if v <= MInf
       status = 'unbounded';
       fprintf( '\n' );
       break;
    end

    % compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

    % possibly plot the trajectory
    if n == 2 && Plotf == 1
       PXY = [ x ,  x - a * g ];
       line( 'XData' , PXY( 1 , : ) , 'YData' , PXY( 2 , : ) , ...
             'LineStyle' , '-' , 'LineWidth' , 2 ,  'Marker' , 'o' , ...
             'Color' , [ 0 0 0 ] );
       drawnow;
    end
    
    x = x - a * g;

    % iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    iter = iter + 1;

    if Interactive
       pause;
    end
end

% end of main loop- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

x = xref;   % return point corresponding to best value found so far

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

end  % the end- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -