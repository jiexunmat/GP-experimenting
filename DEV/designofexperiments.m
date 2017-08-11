function M = designofexperiments(M)
%DESIGNOFEXPERIMENTS   Swordfish design of experiments manager
%
%       M = DESIGNOFEXPERIMENTS(M) populates M.Experiment with new
%       experiments defined in M.ExperimentSetup.DoE(end)
%
%       Author(s): R. Frigola, 16-04-09
%       Copyright (c) 2009 McLaren Racing Limited.
%       $Revision: 1 $  $Date: 16/04/09 19:05 $ $Author: Roger.frigola $  


% We deal with the last defined experiment
iLastDoE = length(M.ExperimentSetup.DoE);

% Find the total number of new experiments. If there are discrete options,
% then the parameter set in MDE is an upper bound.
numNewExperiments = M.ExperimentSetup.DoE(iLastDoE).numOriginalExperiments;
[numActualDesignVariables,NDiscreteOption] = designvariablevectordimension(M);
%with discrete variables, we divide the total number of experiments by the
%number of discrete options
NExperimentsPerOption   = floor(numNewExperiments ./ NDiscreteOption);
numNewExperiments       = ( NExperimentsPerOption * NDiscreteOption );

% Work out dimension of the design variable vector
numVar = length(M.ExperimentSetup.DesignVariableDefinition);

% Work out how many experiments have been defined in the M structure up to
% this point
if ~isfield(M,'Experiment') || isempty(M.Experiment)
    numOldExperiments = 0;
else
    numOldExperiments = length(M.Experiment);
end

% Initialise experiment definition fields that are algorithm-independent
tmp.ID = newguid; % Just to allocate memory
tmp.DesignVariable = [];
ii = 1:length(M.ExperimentSetup.ResponseDefinition);
[tmp.Response(ii).Name] = deal(M.ExperimentSetup.ResponseDefinition(:).Name);
[tmp.Response(ii).Value] = deal([]);
[tmp.Response(ii).XData] = deal([]);
tmp.DerivedResponse = [];
tmp.Status = 'Pending';
tmp.Exceptions = {};
tmp.ExecutionTime = [];
tmp.Worker = '';
[M.Experiment(numOldExperiments+1:numOldExperiments+numNewExperiments)] = deal(tmp);
for i=numOldExperiments+1:numOldExperiments+numNewExperiments
    M.Experiment(i).ID = newguid;
end

switch M.ExperimentSetup.DoE(iLastDoE).Algorithm.Name
    
    case 'MonteCarlo'
        % Random values of the design variables inside its range. Uses a
        % uniform probability distribution.
        
        monteCarloMatrix = rand(numNewExperiments,numActualDesignVariables);
        
        M = swordfishCreateExperiments(M,monteCarloMatrix,numOldExperiments,numNewExperiments,NExperimentsPerOption,numVar,numActualDesignVariables);
        
    case 'LatinHypercubeSampling'
        % Latin Hypercube Sampling: see A. Keane and P. Nair, "Computational
        % Approaches for Aerospace Design, p. 262, Wiley, 2005

        % We first run a normalised sampling
        normalisedLHSampling = lhs(zeros(numActualDesignVariables,1),ones(numActualDesignVariables,1),NExperimentsPerOption);
        
        M = swordfishCreateExperiments(M,normalisedLHSampling,numOldExperiments,numNewExperiments,NExperimentsPerOption,numVar,numActualDesignVariables);
        
    case 'FullFactorial'
        % Generates a regular grid of experiments. Reduces the number of
        % experiments to obtain a regular grid.
        
        % Compute n: the number of levels per dimension
        % We select n in such a way that the number of actual
        % experiments performed is at most numNewExperiments
        n = floor( numNewExperiments^(1/numActualDesignVariables) );
        numUnusedExperiments = numNewExperiments - n ^ numActualDesignVariables;
        numNewExperiments = n ^ numActualDesignVariables;
        
        % Create matrix of grid points
        factorialDesign = FactorialMatrix(n,numActualDesignVariables)
        
        M = swordfishCreateExperiments(M,factorialDesign,numOldExperiments,numNewExperiments,NExperimentsPerOption,numVar,numActualDesignVariables);
        
        % Eliminate unused experiments
        M.Experiment = M.Experiment(1:end-numUnusedExperiments);
        M.ExperimentSetup.DoE(iLastDoE).numOriginalExperiments = numNewExperiments;
        
        
    case 'StarDesign'
        % Generates single design variable sweeps around the baseline.
        
        % Compute numLevels: the number of levels per dimension.
        % We select numLevels in such a way that the number of actual
        % experiments performed is at most numNewExperiments.
        numLevels = floor( (numNewExperiments-1)/numActualDesignVariables + 1 );
        % We also make sure that the number of levels is odd.
        if mod(numLevels,2)==0, numLevels=numLevels-1; end
        tmp = 1 + numActualDesignVariables * (numLevels - 1);
        numUnusedExperiments = numNewExperiments - tmp;
        numNewExperiments = tmp;
        
        % Create matrix of star points
        starDesign = StarMatrix(numLevels,numActualDesignVariables);
        
        % Shuffle points, the idea here is to be more robust if one of the
        % packages goes astray
        [dummy, ii] = sort(rand(numNewExperiments,1));
        starDesign = starDesign(ii,:);
        
        M = swordfishCreateExperiments(M,starDesign,numOldExperiments,numNewExperiments,NExperimentsPerOption,numVar,numActualDesignVariables);
        
        % Eliminate unused experiments
        M.Experiment = M.Experiment(1:end-numUnusedExperiments);
        M.ExperimentSetup.DoE(iLastDoE).numOriginalExperiments = numNewExperiments;
        
    case 'Pareto1'
        % Use already run experiments to create an interim metamodel that
        % is used to compute a Pareto front to decide where to run new
        % experiments
        
        % If no experiments have been previously run, run an lhs with
        % sufficient experiments to overfit a quadratic metamodel and then
        % use the rest of the budget in Pareto1 mode
        
        error('Pareto 1 not yet implemented.')
        
    case 'HighUncertaintyRefinement1'
        % Use already run experiments to create an interim metamodel that
        % has a complete posterior probability distribution. Use this
        % distribution to run new experiments in areas of high uncertainty
        % but taking care that all the new experiments are not bunched up.
        
        % If no experiments have been previously run, run an lhs with
        % sufficient experiments to overfit a quadratic metamodel and then
        % use the rest of the budget in HighUncertaintyRefinement1 mode
        
        error('HighUncertaintyRefinement 1 not yet implemented.')
        
    otherwise
        error('DoE Type not recognised.')
end




%--------------------------------------------------------------------------
%   PRIVATE FUNCTIONS
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function S = lhs(xMin,xMax,nSample)
% LHS  Latin Hypercube Sampling
% Input:
%   xMin    : vector of minimum bounds
%   xMax    : vector of maximum bounds
%   nSample : number of samples
% Output:
%   S       : matrix containing the sample (nSample,numVar)
%
% RF, 20/04/2009

numVar = length(xMin);
ran = rand(nSample,numVar);
S = zeros(nSample,numVar);
for i=1:numVar
   idx = randperm(nSample);
   P = (idx'-ran(:,i))/nSample;
   S(:,i) = xMin(i) + P.* (xMax(i)-xMin(i));
end
S = S(randperm(size(S,1)),:);


%--------------------------------------------------------------------------
function factorialDesign = FactorialMatrix(numLevels,numVar)
% Generates the matrix of a full factorial design with the same number of
% levels for each factor.
%
% RF, 14/05/2009

% No memory preallocation, fix this is the function is ever used to
% generate big matrices
for i=1:numVar
    base = [];
    for j=1:numLevels
        base = [base; j*ones(numLevels^(i-1),1)];
    end
    factorialDesign(:,numVar+1-i) = repmat(base,numLevels^(numVar-i),1);
end

% Normalise
factorialDesign = (factorialDesign-1)/(numLevels-1);


%--------------------------------------------------------------------------
function starDesign = StarMatrix(numLevels,numVar)
% Generates the matrix of a star design with the same number of
% levels for each factor.
% numLevels must be an odd number.
%
% RF, 27/08/2009

% The star centre is at the middle of the range
starDesign(1,:) = 0.5 * ones(1,numVar);

base = 0.5 * ones(numLevels-1,numVar);
starray = 0:1/(numLevels-1):1;
starray = starray([1:(numLevels-1)/2 (numLevels+3)/2:end]);
for i=1:numVar
   ii = 1 + (i-1) * (numLevels-1) + (1:numLevels-1);
   starDesign(ii,:) = base;
   starDesign(ii,i) = starray';
end



%--------------------------------------------------------------------------
function M = swordfishCreateExperiments(M,designMatrix,numOldExperiments,numNewExperiments,NExperimentsPerOption,numVar,numActualDesignVariables)
% SWORDFISHCREATEEXPERIMENTS Creates experiments from a design matrix.
% This function has been factored out from 'FullFactorial'
% With discrete-option design variables, we create a DoE for each option.


%find the indices of any discrete-option design variables
iiDiscreteOption = strcmp({M.ExperimentSetup.DesignVariableDefinition.Type}, 'DiscreteOption');
iiDiscreteOption = find(iiDiscreteOption);
if ~isempty(iiDiscreteOption)
    tempStr = '';
    for i = iiDiscreteOption
        tempStr = [tempStr, mat2str(M.ExperimentSetup.DesignVariableDefinition(i).Range), ','];
    end
    DiscreteOptionMatrix = eval(['combvec(' tempStr(1:end-1) ')']);
end

% For each experiment
for i = numOldExperiments+1 : numOldExperiments+numNewExperiments
    % We keep an index of the elements in the design variable vector
    iDesignVarVector = 1;
    
    % For each variable
    for j = 1:numVar
        M.Experiment(i).DesignVariable(j).Name = M.ExperimentSetup.DesignVariableDefinition(j).Name;
        % Assign design variable value depending on variable type
        switch M.ExperimentSetup.DesignVariableDefinition(j).Type
            case 'Continuous'
                % find the index of the design matrix required
                iDesignMatrix = rem(i-numOldExperiments, NExperimentsPerOption);
                if iDesignMatrix == 0
                    iDesignMatrix = NExperimentsPerOption;
                end
                
                % Random value within range
                varRange = M.ExperimentSetup.DesignVariableDefinition(j).Range;
                M.Experiment(i).DesignVariable(j).Value = varRange(1) + ...
                    designMatrix(iDesignMatrix, iDesignVarVector) * diff(varRange);
                iDesignVarVector = iDesignVarVector + 1;
                
            case 'Discrete'
                % find the index of the design matrix required
                iDesignMatrix = rem(i-numOldExperiments, NExperimentsPerOption);
                if iDesignMatrix == 0
                    iDesignMatrix = NExperimentsPerOption;
                end
                
                % Random integer values between specified range
                varRange = M.ExperimentSetup.DesignVariableDefinition(j).Range;
                M.Experiment(i).DesignVariable(j).Value = round(varRange(1) + ...
                    designMatrix(iDesignMatrix, iDesignVarVector) * diff(varRange));
                iDesignVarVector = iDesignVarVector + 1;
                
            case 'QuasiContinuous'
                % find the index of the design matrix required
                iDesignMatrix = rem(i-numOldExperiments, NExperimentsPerOption);
                if iDesignMatrix == 0
                    iDesignMatrix = NExperimentsPerOption;
                end
                
                % Randomly select one of the alternatives
                map = M.ExperimentSetup.DesignVariableDefinition(j).Map;
                xMap = [map{:,1}];
                range = M.ExperimentSetup.DesignVariableDefinition(j).Range; % Contains min and max values in map
                % Create a sample
                x = range(1) + designMatrix(iDesignMatrix, iDesignVarVector) * diff(range);
                % Find nearest component in map
                ii = nearest(x,xMap,'I');
                % Find component name
                if isempty(map)
                    M.Experiment(i).DesignVariable(j).Value = [];
                else
                    M.Experiment(i).DesignVariable(j).Value = map{ ii , 1};
                end
                iDesignVarVector = iDesignVarVector + 1;
                
            case 'Enumeration'
                % find the index of the design matrix required
                iDesignMatrix = rem(i-numOldExperiments, NExperimentsPerOption);
                if iDesignMatrix == 0
                    iDesignMatrix = NExperimentsPerOption;
                end
                
                % Randomly select one of the alternatives
                map = M.ExperimentSetup.DesignVariableDefinition(j).Map;
                range = M.ExperimentSetup.DesignVariableDefinition(j).Range; % Contains map row indices
                numSteps = length(range);
                iRange = ceil(designMatrix(iDesignMatrix, iDesignVarVector)*numSteps + eps);
                M.Experiment(i).DesignVariable(j).Value = ...
                    map{ range(iRange) , 1};
                iDesignVarVector = iDesignVarVector + 1;
                
            case 'Virtual'
                % find the index of the design matrix required
                iDesignMatrix = rem(i-numOldExperiments, NExperimentsPerOption);
                if iDesignMatrix == 0
                    iDesignMatrix = NExperimentsPerOption;
                end
                
                % We assume that the virtual variables are continuous
                varRange = M.ExperimentSetup.DesignVariableDefinition(j).Range;
                % We deal with the case where some of the
                % dimensions may have zero range
                iiDummyVariables = find( diff(varRange') == 0);
                iiNonDummyVariables = find( diff(varRange') );
                numNonDummyVariables = length(iiNonDummyVariables);
                % Set dummy variables to their fixed value
                if ~isempty(iiDummyVariables)
                    M.Experiment(i).DesignVariable(j).Value(iiDummyVariables) = varRange(iiDummyVariables,1);
                end
                % Set non-dummy variables to their LHS value
                if ~isempty(iiNonDummyVariables)
                    M.Experiment(i).DesignVariable(j).Value(iiNonDummyVariables) = ...
                        varRange(iiNonDummyVariables,1) + ...
                        designMatrix( iDesignMatrix, iDesignVarVector:iDesignVarVector+numNonDummyVariables-1)' .* ...
                        diff(varRange(iiNonDummyVariables,:)')';
                end
                iDesignVarVector = iDesignVarVector + numNonDummyVariables;
                
            case 'DiscreteOption'
                % Here we add the choice for each option, we need to
                % keep track of the cycle of choices.
                iDiscreteOptionIteration = ceil((i-numOldExperiments) ./ NExperimentsPerOption);
                
                for k = 1:length(iiDiscreteOption)
                    if iiDiscreteOption(k) == j
                        M.Experiment(i).DesignVariable(j).Value = DiscreteOptionMatrix(k, iDiscreteOptionIteration);
                        break
                    end
                end
                
                
            otherwise
                error('Type of design variable is not recognised.')
        end
    end
    % Check
    if iDesignVarVector-1 ~= numActualDesignVariables
        error('Something has gone wrong in the Design of Experiments.')
    end
end

