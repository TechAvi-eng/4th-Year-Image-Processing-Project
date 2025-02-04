classdef PointCloudViewerScopeCfg < Simulink.scopes.ScopeBlockSpecification
    %PointCloudViewerScopeCfg   Define the PointCloudViewerScopeCfg class.
    
    %   Copyright 2022-2023 The MathWorks, Inc.
    
    properties (Access = protected)
        Menus
        FileMenuOpeningListener
    end
    
    methods
        
    function this = PointCloudViewerScopeCfg(varargin)
            %PointCloudViewerScopeCfg   Construct the PointCloudViewerScopeCfg class.
            
            % Prevent clear classes warnings
            mlock;
            
            this@Simulink.scopes.ScopeBlockSpecification(varargin{:});
        end
    end
  
    methods (Hidden)
       
        
       function showToolbar = shouldShowMainToolbar(~)
           showToolbar = false;
       end

    end
    methods
  
        function launch(this,makeVisible)
            %launch Creates the scope window.
            
            if nargin<2
                makeVisible = true;
            end
                       
            block = this.Block.Handle;
            mdlRoot = bdroot(block);
            blockName = get_param(block,'Name');
            if strcmp(get_param(mdlRoot,'Lock'),'on')
                errordlg(getString(message('Spcuilib:scopes:ScopeInLockedSystem',strrep(blockName,newline,' '))));
                return
            end
            % Check if we are already launching a ScopeBlock. This can
            % happen when we start the simulation while the Scope is
            % being launched in response to a block open action.
            if ~this.IsLaunching
                this.IsLaunching = true;
                preserve_dirty = Simulink.PreserveDirtyFlag(bdroot(block),'blockDiagram');
                % Check if we need to make a new scope instance.
                if ~isLaunched(this)
                    hFramework = uiscopes.new(this);
                    this.Block.UnifiedScope = hFramework; % Now Scope becomes launched
                    if strcmpi(get_param(mdlRoot,'BlockDiagramType'),'library')
                        % Simulation controls should not be shown if block is inside a
                        % library.
                        this.RenderSimulationControls = false;
                    end
                    % Fix Scope Position
                    hFigure = hFramework.Parent;
                    % Make sure that we have everything set up properly.
                    setScopePosition(this);
                    if ~strcmp(get(hFigure,'WindowStyle'),'docked')
                        set(hFigure,'Position',uiservices.fixFigurePosition(get(hFigure,'Position')));
                    end
                    setScopeParams(this);
                    attachListeners(this,hFramework);
                end
                % If the data source is already set up, do not reconnect.
                if (isempty(hFramework.DataSource) || hFramework.DataSource.BlockHandle~=this.Block)
                    connectDataSource( this, hFramework );
                end
                if makeVisible
                    visible(hFramework,'on');
                    % Remove the preShow callback as Scope is now launched
                    removePreShowCallback( this );
                end
                
                % At this point the display (plotters) should also have all
                % the information about the source.
                this.IsLaunching = false;
                delete( preserve_dirty );
            end
        end 
        
              
        function removePreShowCallback(this)
            if this.Block.PreShowCallbackExists
                mdlRoot = bdroot(this.Block.Handle);
                obj = get_param(mdlRoot,'Object');
                obj.removeCallback('PreShow',['Scope',num2hex(this.Block.Handle)]);
                this.Block.PreShowCallbackExists = false;
            end      
        end
          
        function fixPositionAndScopeParams(this)
            hFramework = this.Block.UnifiedScope;
            hFigure = hFramework.Parent;
            % Make sure that we have everything set up properly.
            setScopePosition( this );
            if ~strcmp( get( hFigure, 'WindowStyle' ), 'docked' )
                set( hFigure, 'Position', uiservices.fixFigurePosition( get( hFigure, 'Position' ) ) );
            end
            setScopeParams( this );
        end
             
        function connectDataSource(this,hFramework)
            if nargin<2
                hFramework = this.Block.UnifiedScope;
            end
            % Simulation could have started when the Scope is being
            % opened. In this case, if the runtime block is empty,
            % repopulate it by obtaining from the block.

            blockHandle = this.Block.Handle;
            rto = get_param(blockHandle,'RunTimeObject');
            blockObj = get_param(blockHandle,'object');
            
            connectToWiredSL1( hFramework, { blockObj, rto } );
            connectToWiredSL2( hFramework );
            connectToWiredSL3( hFramework );
            connectToWiredSL4( hFramework );
            connectToWiredSL5( hFramework );

        end

        function b = isNormalMode(this)
            mdlRoot = bdroot( this.Block.Handle );
            b = strcmp( get_param( mdlRoot, 'SimulationMode' ), 'normal' );
        end
        
        function setUpdateMethod(this,install)
            if ~isempty(this.Block)
                block = this.Block.Handle;
                if (nargin==2)
                    if (install==false)
                        oc = '0';  % closed
                    else
                        oc = '1';  % open
                    end
                    set_param(block,'FigureOpen',oc);
                end
            end
        end
 
        function appName = getScopeTag(~)
            %getAppName Returns the simple application name.
            appName = 'Point Cloud Viewer';
        end
        
        function setScopeParams(this)
            % Only set the scope parameters when the scope is launched and
            % we are allowing scope parameter changes.  We need to make
            % sure that the block does not try to update all of the scope
            % settings at once while we're setting a single parameter.
            
            if ~isLaunched( this )  || ~this.AllowScopeChanges
                return
            end
            
            hBlock = this.Block;
            hScope = this.Block.ScopeSpecificationObject;

            specRange = get_param(hBlock.Handle,'specRange');
            setScopeParam(hScope, 'Visuals', 'Point Cloud', ...
                'UseDataRange', strcmp(specRange, 'on'));
            
            minInputVal = get_param(hBlock.Handle,'minInputVal');
            setScopeParam(hScope, 'Visuals', 'Point Cloud', ...
                'DataRangeMin', evalin('base', minInputVal));
            
            maxInputVal = get_param(hBlock.Handle,'maxInputVal');
            setScopeParam(hScope, 'Visuals', 'Point Cloud', ...
                'DataRangeMax', evalin('base', maxInputVal));
        end
       
        function h = getWidget(this,varargin)
            %getWidget Returns the handle to a UIMGR widget.
            %   getWidget(H, PATH1, PATH2, etc.) returns the handle to a
            %   UIMGR widget specified by the path.
            if isLaunched( this )
                h = findobj( this.Block.UnifiedScope, 'tag', varargin{ : } );
            else
                h = [  ];
            end
        end
        
        function setScopeParam(this,type,name,propName,value)
            %setScopeParam Set the scope parameter.
            %   setScopeParam(H, TYPE, NAME, PROPNAME, VALUE) Set the scope
            %   parameter PROPNAME to VALUE for the extension specified by
            %   TYPE and NAME.
            
            %   Copyright 2022 The MathWorks, Inc.
            
            if ~isLaunched(this) || ~this.AllowScopeChanges
                return
            end
            cfgDb = this.Block.UnifiedScope.ExtDriver.ConfigurationSet;
            % Get the specified config object.
            cfg = cfgDb.findConfig(type, name);
            % Get the specified property object.
            setValue(cfg.PropertySet, propName, value);
            
        end
                
        function setBlockParams(this, ed)
            hScope = this.Block.ScopeSpecificationObject;
            
            this.AllowScopeChanges = false;
            
            try
                if nargin<2 || shouldDirtyModel( ed )
                    bdObj = get_param( bdroot( this.Block.Handle ), 'Object' );
                    bdObj.setDirty( 'blockDiagram', true );
                end

                if getScopeParam(this, 'Visuals', 'Point Cloud', 'UseDataRange')
                    specRange = 'on';
                else
                    specRange = 'off';
                end
                
                minInputVal = mat2str(getScopeParam(hScope, 'Visuals', 'Point Cloud', 'DataRangeMin'));
                maxInputVal = mat2str(getScopeParam(hScope, 'Visuals', 'Point Cloud', 'DataRangeMax'));
                setBlockParam(this, 'specRange', specRange);
                setBlockParam(this, 'minInputVal', minInputVal);
                setBlockParam(this, 'maxInputVal', maxInputVal);
            catch E
                this.AllowScopeChanges = true;
                if strcmp( E.identifier, 'Simulink:Engine:CannotChangeConstTsBlks' )
                    return
                end
                rethrow( E );                
            end
            this.AllowScopeChanges = true;
            function b = shouldDirtyModel(ed)
                if isempty( ed )
                    b = true;
                elseif isprop( ed, 'Data' )
                    % If the event was not user generated then the model
                    % should not be dirtied
                    b = ~isfield( ed.Data, 'UserGenerated' ) || ed.Data.UserGenerated;
                elseif isprop( ed, 'AffectedObject' )
                    % If the event comes from PreviousZoomMode or
                    % PreviousAutoscale the model should not be dirtied.
                    b = ~isprop( ed.AffectedObject, 'Name' ) || ~any( strcmp( ed.AffectedObject.Name, { 'PreviousZoomMode', 'PreviousAutoscale' } ) );
                else
                    b = true;
                end
            end
        end
        
        function mdlStart(this)   
            hScope = this.Block.ScopeSpecificationObject;

            set(hScope.getWidget('Base/Menus/File/Sources/ImageSignal'), ...
                'Enable', 'off');
            set(hScope.getWidget('Base/Menus/File/Sources/OpenAtMdlStart'), ...
                'Enable', 'off');
        end
        
        function show = showPrintAction(~, type)
            %showPrintAction Returns true to add Printing menus and toolbars in
            % uiscopes.Framework. If printing functionality is not available in
            % particular scope application, the subclass should override this
            % method to return false.
            show = strcmp(type, 'menu');
        end
        
        function mdlTerminate(this)

             hScope = this.Block.ScopeSpecificationObject;
             set(hScope.getWidget('Base/Menus/File/Sources/ImageSignal'), ...
                 'Enable', 'on');
             set(hScope.getWidget('Base/Menus/File/Sources/OpenAtMdlStart'), ...
                 'Enable', 'on');
        end
        
        function b = getOpenAtMdlStart(this)
            b = strcmp(get_param(this.Block.Handle,'OpenAtMdlStart'), 'on');
        end

        function b = useBlockInterface(~)
             b = true;
        end
        
        function b = showSaveConfiguration(~)
            %showSaveConfiguration
            b = true;
        end
        
        function renderMenus(this, hScope)
            this.FileMenuOpeningListener = event.listener(hScope, ...
                'FileMenuOpening', @this.onFileMenuOpening);
        end
        
        function b = showKeyboardCommand(~)
            %showKeyboardCommand - Returns true when the keyboard
            %command help menu item should be shown
            b = false;
        end
        
        function [mApp, mExample, mAbout] = createHelpMenuItems(~, mHelp)            
            mApp(1) = uimenu(mHelp, ...
                'Tag', 'uimgr.uimenu_PointCloudViewerHelp', ...
                'Label', 'Point Cloud Viewer &Help', ...
                'Callback', @(hco,ev) helpview('vision', 'visionpointcloudviewer'));
            
            mApp(2) = uimenu(mHelp, ...
                'Tag', 'uimgr.uimenu_VIPBlks', ...
                'Label', '&Computer Vision Toolbox Help', ...
                'Callback', @(hco,ev) helpview('vision', 'visioninfo'));
            
            mExample = uimenu(mHelp, ...
                'Tag', 'uimgr.uimenu_VIPBlks Demos', ...
                'Label', 'Computer Vision Toolbox &Examples', ...
                'Callback', @(hco,ev) visiondemos);
            
            % Want the "About" option separated, so we group everything above
            % into a menugroup and leave "About" as a singleton menu
            mAbout = uimenu(mHelp, ...
                'Tag', 'uimgr.uimenu_About', ...
                'Label', '&About Computer Vision Toolbox', ...
                'Callback', @(hco,ev) aboutvipblks);
        end
        
        function cfgFile = getConfigurationFile(~)
            cfgFile = 'pointcloudviewer.cfg';
        end
        
        function className = getConfigurationClass(~)
            className = 'vipscopes.PointCloudViewerConfiguration';
        end
        
        function helpArgs = getHelpArgs(~, key)
            if nargin < 2
                helpArgs = {'vippointcloudviewer'};
            else                               
                switch lower(key)
                    case 'colormap'
                        helpArgs = {'uiservices.helpview', 'vision', 'video_viewer_colormap'};
                    case 'overall'
                        helpArgs = {'vippointcloudviewer'};
                    otherwise
                        helpArgs = {};
                end
            end
        end
        
        function setScopePosition(this)
            %setScopePosition Set the scope position into the figure.
            
            % If there is no framework, then we cannot set the scope
            % position, simply return.  If we are docked, a warning will
            % occur when we try to set the scope position.
            hFramework = this.Block.UnifiedScope;
            if isempty(hFramework) || ...
                    strcmp(get(hFramework.Parent, 'WindowStyle'), 'docked')
                return;
            end
            
            hBlock = this.Block;
            figPos = get_param(hBlock.Handle,'FigPos');
            figPosHG = evalin('base', figPos);

            % If we are loading in an old block (pre7b), it will be in HG
            % format already.
            
            if strcmp(get_param(hBlock.Handle, 'inputType'), 'Obsolete7b')
                figPosHG = togglePositionFormat(figPosHG);
            end
            set(hFramework.Parent, 'Position', figPosHG);
        end
        
        function saveScopePosition(this, figPosHG)
            %saveScopePosition Save the position of the scope into the
            %block.
            
            hBlock = this.Block;
            
            % Set the Figure Position. Convert from FigPos, which is from
            % the northwest corner, to HG position, which is from the
            % southwest corner.
            figPos = mat2str(togglePositionFormat(figPosHG));
            set_param(hBlock.Handle,'FigPos',figPos);
        end
        
        function hiddenExts = getHiddenExtensions(~)
            hiddenExts = {};
        end

        function hiddenTypes = getHiddenTypes(~)
            hiddenTypes = {'Sources', 'Visuals', 'Tools'};
        end
        
        function numInputs = getNumInputs(this)
            type = get(this.Block.Handle, 'imagePorts');
            if strcmp(type, 'Location Port')
                numInputs = 1;
            else
                numInputs = 2;
            end
        end
        
        function attachListeners(this,hFramework)
            % Add a listener to the scope's closing event.
            this.ScopeParamsListener = createPropertyListener(hFramework,@(h,ed) setBlockParams( this,ed));
            this.ScopeExtensionListener = createExtensionListener(hFramework,@this.extensionAddedOrRemoved); 
            this.ScopeCloseListener = addlistener(hFramework,'Close',@this.scopeCloseCallback);
            hFig = hFramework.Parent;

            this.ScopeVisibleListener = addlistener(hFig,hFig.findprop('Visible'), ...
                'PostSet',@this.onScopeVisibleChange);
            
            % Add a listener to set the LockSynchronous property on
            % the block if it changes on the source
            this.LockSynchronousListener = addlistener(hFramework,'SynchronousLockChanged',@this.onLockSynchronousChange);
        end

        function setBlockParam(this,paramName,paramValue)
            %setBlockParam Set the block parameter
            %   setBlockParam(h, ParamName, ParamValue)
            % Check codesearchif we are in a state that allows block changes before
            % proceeding.  This usually happens when we are changing a
            % scope parameter and do not want to double set.

            try
                hBlock = this.Block.Handle;
                %Check if the scope block is not inside a locked library.
                %We should not be changing block parameters if the block is
                %inside a locked library.
                if strcmp( get_param( bdroot( hBlock ), 'Lock' ), 'off' ) || strcmp( get_param( hBlock, 'LinkStatus' ), 'implicit' )
                    set_param( hBlock, paramName, paramValue );
                end
            catch E
                if strcmp( E.identifier, 'Simulink:Engine:CannotChangeConstTsBlks' )
                    return
                end
                rethrow( E );
            end
        end
        
        function b = useMCOSExtMgr(~)
            b = true;
        end
        
        function b = useUIMgr(~)
            b = false;
        end
        
        function b = needsMenuGroups(~)
            b = true;
        end
        
        function b = getNeedRuntimeCallbacks(~)
            b = false;
        end
        
        function b = useStagedConstruction(~)
            b = false;
        end
        
    end
    
    methods(Access = protected)
        
        function onFileMenuOpening(this, hScope, ~)
            
            menus = this.Menus;
            if isempty(menus)
                hFileMenu = findobj(hScope.Parent, 'Tag', 'uimgr.uimenugroup_File');
                
                menus.openAtModelStart = uimenu(hFileMenu, ...
                    'Label', 'Open at Start of Simulation', ...
                    'Position', 1, ...
                    'Tag', 'uimgr.spctogglemenu_OpenAtMdlStart', ...
                    'Callback', @(~, ~) toggleOpenAtMdlStart(this));
                menus.onePort = uimenu(hFileMenu, ...
                    'Position', 2, ...
                    'Separator', 'on', ...
                    'Tag', 'uimgr.spctogglemenu_MultiImagePort', ...
                    'Label', 'Location Port', ...
                    'Callback', @(h, ev) set_param(this.Block.Handle, 'imagePorts', 'Location Port'));
                menus.twoPorts = uimenu(hFileMenu, ...
                    'Position', 3, ...
                    'Tag', 'uimgr.spctogglemenu_OneImagePort', ...
                    'Label', 'Location and Color Port', ...
                    'Callback', @(h, ev) set_param(this.Block.Handle, 'imagePorts', 'Location and Color Port'));
                
                this.Menus = menus;
            end
            
            imagePorts = get_param(this.Block.Handle,'imagePorts');
            if strcmp(imagePorts, 'Location Port')
                oneChecked   = 'on';
                twoChecked = 'off';
            else
                oneChecked   = 'off';
                twoChecked = 'on';
            end
            if strcmp(get(bdroot(this.Block.Handle), 'SimulationStatus'), 'stopped')
                enabState = 'on';
            else
                enabState = 'off';
            end
            set(menus.onePort, ...
                'Checked', oneChecked, ...
                'Enable',  enabState);
            set(menus.twoPorts, ...
                'Checked', twoChecked, ...
                'Enable', enabState);
            set(menus.openAtModelStart, ...
                'Checked', get(this.Block.Handle, 'OpenAtMdlStart'));
            
        end
        
        % cache config params so they are not need to be repeatedly loaded
        function retVal = getDefaultConfigParams(this)
            
            persistent defaultConfigParams;
            
            if isempty(defaultConfigParams) 
                defaultConfigParams = extmgr.ConfigurationSet.createAndLoad(this.getConfigurationFile);
            end
            
            retVal = defaultConfigParams;
        end
    end
    
    methods (Static, Hidden)
        
        function this = loadobj(s)
            this = loadobj@ Simulink.scopes.ScopeBlockSpecification(s);
            if isempty(this.CurrentConfiguration)
                return;
            end
            cfg = this.CurrentConfiguration.findConfig('Core', 'Source UI');
            if isempty(cfg) || isempty(cfg.PropertyDb)
                return;
            end
            prop = cfg.PropertyDb.findProp('ShowPlaybackCmdMode');
            if isempty(prop)
                return;
            end
            prop.Value = false;
        end
    end
end

% -------------------------------------------------------------------------
function toggleOpenAtMdlStart(this)

if strcmp(get_param(this.Block.Handle, 'OpenAtMdlStart'), 'on')
    newValue = 'off';
else
    newValue = 'on';
end

set_param(this.Block.Handle, 'OpenAtMdlStart', newValue);
end

% -------------------------------------------------------------------------
function figPos = togglePositionFormat(figPos)
% Convert between windows and HG positioning.  The same calculation
% reverses itself.

origUnits = get(0, 'units');
set(0, 'units', 'pix');
screenSize = get(0, 'screenSize'); % [left bottom width height] in pixels
set(0,'units', origUnits);   % restore the resolution settings

figPos(2) = screenSize(4) - figPos(2) - 1;
end

% [EOF]
