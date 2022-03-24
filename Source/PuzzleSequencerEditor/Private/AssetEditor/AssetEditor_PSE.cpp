#include "AssetEditor/AssetEditor_PSE.h"
#include "AssetEditor/AssetEditorToolbar_PSE.h"
#include "AssetEditor/AssetGraphSchema_PSE.h"
#include "AssetEditor/EditorCommands_PSE.h"
#include "AssetEditor/EdGraph_PSE.h"
#include "AssetEditor/EdNode_PSENode.h"
#include "AssetEditor/EdNode_PSEEdge.h"
#include "AssetToolsModule.h"
#include "HAL/PlatformApplicationMisc.h"
#include "Framework/Commands/GenericCommands.h"
#include "GraphEditorActions.h"
#include "IDetailsView.h"
#include "PropertyEditorModule.h"
#include "EdGraphUtilities.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "UObject/ObjectSaveContext.h"

#define LOCTEXT_NAMESPACE "AssetEditor_PuzzleSequencer"

const FName PuzzleSequencerEditorAppName = FName(TEXT("PuzzleSequencerEditorApp"));

struct FPSEAssetEditorTabs
{
	// Tab identifiers
	static const FName PuzzleSequencerPropertyID;
	static const FName ViewportID;
	static const FName PuzzleSequencerEditorSettingsID;
};

//////////////////////////////////////////////////////////////////////////

const FName FPSEAssetEditorTabs::PuzzleSequencerPropertyID(TEXT("PuzzleSequencerProperty"));
const FName FPSEAssetEditorTabs::ViewportID(TEXT("Viewport"));
const FName FPSEAssetEditorTabs::PuzzleSequencerEditorSettingsID(TEXT("PuzzleSequencerEditorSettings"));

//////////////////////////////////////////////////////////////////////////

FAssetEditor_PSE::FAssetEditor_PSE()
{
	EditorSettings = NewObject<USettings_PSE>(USettings_PSE::StaticClass());
	OnPackageSavedDelegateHandle = UPackage::PackageSavedWithContextEvent.AddRaw(this, &FAssetEditor_PSE::OnPackageSavedWithContext);
}

FAssetEditor_PSE::~FAssetEditor_PSE()
{
	UPackage::PackageSavedWithContextEvent.Remove(OnPackageSavedDelegateHandle);
}

void FAssetEditor_PSE::InitPuzzleSequencerAssetEditor(const EToolkitMode::Type InMode, const TSharedPtr<IToolkitHost>& InInitToolkitHost, UPuzzleSequencer* InGraph)
{
	EditingGraph = InGraph;
	CreateEdGraph();

	FGenericCommands::Register();
	FGraphEditorCommands::Register();
	FEditorCommands_PSE::Register();

	if (!ToolbarBuilder.IsValid())
	{
		ToolbarBuilder = MakeShareable(new FAssetEditorToolbar_PSE(SharedThis(this)));
	}

	BindCommands();

	CreateInternalWidgets();

	TSharedPtr<FExtender> ToolbarExtender = MakeShareable(new FExtender);

	ToolbarBuilder->AddPuzzleSequencerToolbar(ToolbarExtender);

	// Layout
	const TSharedRef<FTabManager::FLayout> StandaloneDefaultLayout = FTabManager::NewLayout("Standalone_PuzzleSequencerEditor_Layout_v1")
		->AddArea
		(
			FTabManager::NewPrimaryArea()->SetOrientation(Orient_Vertical)
			                             ->Split
			                             (
				                             FTabManager::NewStack()
				                             ->SetSizeCoefficient(0.1f)
				                             //				                             ->AddTab(GetToolbarTabId(), ETabState::OpenedTab)->SetHideTabWell(true)
			                             )
			                             ->Split
			                             (
				                             FTabManager::NewSplitter()->SetOrientation(Orient_Horizontal)->SetSizeCoefficient(0.9f)
				                                                       ->Split
				                                                       (
					                                                       FTabManager::NewStack()
					                                                       ->SetSizeCoefficient(0.65f)
					                                                       ->AddTab(FPSEAssetEditorTabs::ViewportID, ETabState::OpenedTab)->SetHideTabWell(true)
				                                                       )
				                                                       ->Split
				                                                       (
					                                                       FTabManager::NewSplitter()->SetOrientation(Orient_Vertical)
					                                                                                 ->Split
					                                                                                 (
						                                                                                 FTabManager::NewStack()
						                                                                                 ->SetSizeCoefficient(0.7f)
						                                                                                 ->AddTab(FPSEAssetEditorTabs::PuzzleSequencerPropertyID, ETabState::OpenedTab)->SetHideTabWell(true)
					                                                                                 )
					                                                                                 ->Split
					                                                                                 (
						                                                                                 FTabManager::NewStack()
						                                                                                 ->SetSizeCoefficient(0.3f)
						                                                                                 ->AddTab(FPSEAssetEditorTabs::PuzzleSequencerEditorSettingsID, ETabState::OpenedTab)
					                                                                                 )
				                                                       )
			                             )
		);

	const bool bCreateDefaultStandaloneMenu = true;
	const bool bCreateDefaultToolbar = true;
	FAssetEditorToolkit::InitAssetEditor(InMode, InInitToolkitHost, PuzzleSequencerEditorAppName, StandaloneDefaultLayout, bCreateDefaultStandaloneMenu, bCreateDefaultToolbar, EditingGraph, false);

	RegenerateMenusAndToolbars();
}

void FAssetEditor_PSE::RegisterTabSpawners(const TSharedRef<FTabManager>& InTabManager)
{
	WorkspaceMenuCategory = InTabManager->AddLocalWorkspaceMenuCategory(LOCTEXT("WorkspaceMenu_PuzzleSequencerEditor", "Puzzle Sequencer Editor"));
	auto WorkspaceMenuCategoryRef = WorkspaceMenuCategory.ToSharedRef();

	FAssetEditorToolkit::RegisterTabSpawners(InTabManager);

	InTabManager->RegisterTabSpawner(FPSEAssetEditorTabs::ViewportID, FOnSpawnTab::CreateSP(this, &FAssetEditor_PSE::SpawnTab_Viewport))
	            .SetDisplayName(LOCTEXT("GraphCanvasTab", "Viewport"))
	            .SetGroup(WorkspaceMenuCategoryRef)
	            .SetIcon(FSlateIcon(FEditorStyle::GetStyleSetName(), "GraphEditor.EventGraph_16x"));

	InTabManager->RegisterTabSpawner(FPSEAssetEditorTabs::PuzzleSequencerPropertyID, FOnSpawnTab::CreateSP(this, &FAssetEditor_PSE::SpawnTab_Details))
	            .SetDisplayName(LOCTEXT("DetailsTab", "Property"))
	            .SetGroup(WorkspaceMenuCategoryRef)
	            .SetIcon(FSlateIcon(FEditorStyle::GetStyleSetName(), "LevelEditor.Tabs.Details"));

	InTabManager->RegisterTabSpawner(FPSEAssetEditorTabs::PuzzleSequencerEditorSettingsID, FOnSpawnTab::CreateSP(this, &FAssetEditor_PSE::SpawnTab_EditorSettings))
	            .SetDisplayName(LOCTEXT("EditorSettingsTab", "Puzzle Sequencer Editor Setttings"))
	            .SetGroup(WorkspaceMenuCategoryRef)
	            .SetIcon(FSlateIcon(FEditorStyle::GetStyleSetName(), "LevelEditor.Tabs.Details"));
}

void FAssetEditor_PSE::UnregisterTabSpawners(const TSharedRef<FTabManager>& InTabManager)
{
	FAssetEditorToolkit::UnregisterTabSpawners(InTabManager);

	InTabManager->UnregisterTabSpawner(FPSEAssetEditorTabs::ViewportID);
	InTabManager->UnregisterTabSpawner(FPSEAssetEditorTabs::PuzzleSequencerPropertyID);
	InTabManager->UnregisterTabSpawner(FPSEAssetEditorTabs::PuzzleSequencerEditorSettingsID);
}

FName FAssetEditor_PSE::GetToolkitFName() const
{
	return FName("FPuzzleSequencerEditor");
}

FText FAssetEditor_PSE::GetBaseToolkitName() const
{
	return LOCTEXT("PuzzleSequencerEditorAppLabel", "Puzzle Sequencer Editor");
}

FText FAssetEditor_PSE::GetToolkitName() const
{
	const bool bDirtyState = EditingGraph->GetOutermost()->IsDirty();

	FFormatNamedArguments Args;
	Args.Add(TEXT("PuzzleSequencerName"), FText::FromString(EditingGraph->GetName()));
	Args.Add(TEXT("DirtyState"), bDirtyState ? FText::FromString(TEXT("*")) : FText::GetEmpty());
	return FText::Format(LOCTEXT("PuzzleSequencerEditorToolkitName", "{PuzzleSequencerName}{DirtyState}"), Args);
}

FText FAssetEditor_PSE::GetToolkitToolTipText() const
{
	return FAssetEditorToolkit::GetToolTipTextForObject(EditingGraph);
}

FLinearColor FAssetEditor_PSE::GetWorldCentricTabColorScale() const
{
	return FLinearColor::White;
}

FString FAssetEditor_PSE::GetWorldCentricTabPrefix() const
{
	return TEXT("PuzzleSequencerEditor");
}

FString FAssetEditor_PSE::GetDocumentationLink() const
{
	return TEXT("");
}

void FAssetEditor_PSE::SaveAsset_Execute()
{
	if (EditingGraph != nullptr)
	{
		RebuildGraph();
	}

	FAssetEditorToolkit::SaveAsset_Execute();
}

void FAssetEditor_PSE::UpdateToolbar()
{
}

void FAssetEditor_PSE::RegisterToolBarTab(const TSharedRef<FTabManager>& InTabManager)
{
	FAssetEditorToolkit::RegisterTabSpawners(InTabManager);
}

void FAssetEditor_PSE::AddReferencedObjects(FReferenceCollector& Collector)
{
	Collector.AddReferencedObject(EditingGraph);
	Collector.AddReferencedObject(EditingGraph->EdGraph);
}

USettings_PSE* FAssetEditor_PSE::GetSettings() const
{
	return EditorSettings;
}

FString FAssetEditor_PSE::GetReferencerName() const
{
	return TEXT("AssetEditor_PSE");
}

TSharedRef<SDockTab> FAssetEditor_PSE::SpawnTab_Viewport(const FSpawnTabArgs& Args)
{
	check(Args.GetTabId() == FPSEAssetEditorTabs::ViewportID);

	TSharedRef<SDockTab> SpawnedTab = SNew(SDockTab)
		.Label(LOCTEXT("ViewportTab_Title", "Viewport"));

	if (ViewportWidget.IsValid())
	{
		SpawnedTab->SetContent(ViewportWidget.ToSharedRef());
	}

	return SpawnedTab;
}

TSharedRef<SDockTab> FAssetEditor_PSE::SpawnTab_Details(const FSpawnTabArgs& Args)
{
	check(Args.GetTabId() == FPSEAssetEditorTabs::PuzzleSequencerPropertyID);

	return SNew(SDockTab)
		.Label(LOCTEXT("Details_Title", "Property"))
	[
		PropertyWidget.ToSharedRef()
	];
}

TSharedRef<SDockTab> FAssetEditor_PSE::SpawnTab_EditorSettings(const FSpawnTabArgs& Args)
{
	check(Args.GetTabId() == FPSEAssetEditorTabs::PuzzleSequencerEditorSettingsID);

	return SNew(SDockTab)
		.Label(LOCTEXT("EditorSettings_Title", "Puzzle Sequencer Editor Setttings"))
	[
		EditorSettingsWidget.ToSharedRef()
	];
}

void FAssetEditor_PSE::CreateInternalWidgets()
{
	ViewportWidget = CreateViewportWidget();

	FDetailsViewArgs Args;
	Args.bHideSelectionTip = true;
	Args.NotifyHook = this;

	FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
	PropertyWidget = PropertyModule.CreateDetailView(Args);
	PropertyWidget->SetObject(EditingGraph);
	PropertyWidget->OnFinishedChangingProperties().AddSP(this, &FAssetEditor_PSE::OnFinishedChangingProperties);

	EditorSettingsWidget = PropertyModule.CreateDetailView(Args);
	EditorSettingsWidget->SetObject(EditorSettings);
}

TSharedRef<SGraphEditor> FAssetEditor_PSE::CreateViewportWidget()
{
	FGraphAppearanceInfo AppearanceInfo;
	AppearanceInfo.CornerText = LOCTEXT("AppearanceCornerText_PuzzleSequencer", "Puzzle Sequencer");

	CreateCommandList();

	SGraphEditor::FGraphEditorEvents InEvents;
	InEvents.OnSelectionChanged = SGraphEditor::FOnSelectionChanged::CreateSP(this, &FAssetEditor_PSE::OnSelectedNodesChanged);
	InEvents.OnNodeDoubleClicked = FSingleNodeEvent::CreateSP(this, &FAssetEditor_PSE::OnNodeDoubleClicked);

	return SNew(SGraphEditor)
		.AdditionalCommands(GraphEditorCommands)
		.IsEditable(true)
		.Appearance(AppearanceInfo)
		.GraphToEdit(EditingGraph->EdGraph)
		.GraphEvents(InEvents)
		.AutoExpandActionMenu(true)
		.ShowGraphStateOverlay(false);
}

void FAssetEditor_PSE::BindCommands()
{
	ToolkitCommands->MapAction(FEditorCommands_PSE::Get().GraphSettings,
	                           FExecuteAction::CreateSP(this, &FAssetEditor_PSE::GraphSettings),
	                           FCanExecuteAction::CreateSP(this, &FAssetEditor_PSE::CanGraphSettings)
	);
}

void FAssetEditor_PSE::CreateEdGraph()
{
	if (EditingGraph->EdGraph == nullptr)
	{
		EditingGraph->EdGraph = CastChecked<UEdGraph_PSE>(FBlueprintEditorUtils::CreateNewGraph(EditingGraph, NAME_None, UEdGraph_PSE::StaticClass(), UAssetGraphSchema_PSE::StaticClass()));
		EditingGraph->EdGraph->bAllowDeletion = false;

		// Give the schema a chance to fill out any required nodes (like the results node)
		const UEdGraphSchema* Schema = EditingGraph->EdGraph->GetSchema();
		Schema->CreateDefaultNodesForGraph(*EditingGraph->EdGraph);
	}
}

void FAssetEditor_PSE::CreateCommandList()
{
	if (GraphEditorCommands.IsValid())
	{
		return;
	}

	GraphEditorCommands = MakeShareable(new FUICommandList);

	// Can't use CreateSP here because derived editor are already implementing TSharedFromThis<FAssetEditorToolkit>
	// however it should be safe, since commands are being used only within this editor
	// if it ever crashes, this function will have to go away and be reimplemented in each derived class

	GraphEditorCommands->MapAction(FEditorCommands_PSE::Get().GraphSettings,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::GraphSettings),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanGraphSettings));

	GraphEditorCommands->MapAction(FGenericCommands::Get().SelectAll,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::SelectAllNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanSelectAllNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Delete,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::DeleteSelectedNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanDeleteNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Copy,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CopySelectedNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanCopyNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Cut,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CutSelectedNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanCutNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Paste,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::PasteNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanPasteNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Duplicate,
	                               FExecuteAction::CreateRaw(this, &FAssetEditor_PSE::DuplicateNodes),
	                               FCanExecuteAction::CreateRaw(this, &FAssetEditor_PSE::CanDuplicateNodes)
	);

	GraphEditorCommands->MapAction(FGenericCommands::Get().Rename,
	                               FExecuteAction::CreateSP(this, &FAssetEditor_PSE::OnRenameNode),
	                               FCanExecuteAction::CreateSP(this, &FAssetEditor_PSE::CanRenameNodes)
	);
}

TSharedPtr<SGraphEditor> FAssetEditor_PSE::GetCurrGraphEditor() const
{
	return ViewportWidget;
}

FGraphPanelSelectionSet FAssetEditor_PSE::GetSelectedNodes() const
{
	FGraphPanelSelectionSet CurrentSelection;
	TSharedPtr<SGraphEditor> FocusedGraphEd = GetCurrGraphEditor();
	if (FocusedGraphEd.IsValid())
	{
		CurrentSelection = FocusedGraphEd->GetSelectedNodes();
	}

	return CurrentSelection;
}

void FAssetEditor_PSE::RebuildGraph()
{
	if (EditingGraph == nullptr)
	{
		return;
	}

	UEdGraph_PSE* EdGraph = Cast<UEdGraph_PSE>(EditingGraph->EdGraph);
	check(EdGraph != nullptr);

	EdGraph->RebuildGraph();
}

void FAssetEditor_PSE::SelectAllNodes()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (CurrentGraphEditor.IsValid())
	{
		CurrentGraphEditor->SelectAllNodes();
	}
}

bool FAssetEditor_PSE::CanSelectAllNodes()
{
	return true;
}

void FAssetEditor_PSE::DeleteSelectedNodes()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (!CurrentGraphEditor.IsValid())
	{
		return;
	}

	const FScopedTransaction Transaction(FGenericCommands::Get().Delete->GetDescription());

	CurrentGraphEditor->GetCurrentGraph()->Modify();

	const FGraphPanelSelectionSet SelectedNodes = CurrentGraphEditor->GetSelectedNodes();
	CurrentGraphEditor->ClearSelectionSet();

	for (FGraphPanelSelectionSet::TConstIterator NodeIt(SelectedNodes); NodeIt; ++NodeIt)
	{
		UEdGraphNode* EdNode = Cast<UEdGraphNode>(*NodeIt);
		if (EdNode == nullptr || !EdNode->CanUserDeleteNode())
		{
			continue;
		};

		if (UEdNode_PSENode* EdNode_Node = Cast<UEdNode_PSENode>(EdNode))
		{
			EdNode_Node->Modify();

			const UEdGraphSchema* Schema = EdNode_Node->GetSchema();
			if (Schema != nullptr)
			{
				Schema->BreakNodeLinks(*EdNode_Node);
			}

			EdNode_Node->DestroyNode();
		}
		else
		{
			EdNode->Modify();
			EdNode->DestroyNode();
		}
	}
}

bool FAssetEditor_PSE::CanDeleteNodes()
{
	// If any of the nodes can be deleted then we should allow deleting
	const FGraphPanelSelectionSet SelectedNodes = GetSelectedNodes();
	for (FGraphPanelSelectionSet::TConstIterator SelectedIter(SelectedNodes); SelectedIter; ++SelectedIter)
	{
		UEdGraphNode* Node = Cast<UEdGraphNode>(*SelectedIter);
		if (Node != nullptr && Node->CanUserDeleteNode())
		{
			return true;
		}
	}

	return false;
}

void FAssetEditor_PSE::DeleteSelectedDuplicatableNodes()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (!CurrentGraphEditor.IsValid())
	{
		return;
	}

	const FGraphPanelSelectionSet OldSelectedNodes = CurrentGraphEditor->GetSelectedNodes();
	CurrentGraphEditor->ClearSelectionSet();

	for (FGraphPanelSelectionSet::TConstIterator SelectedIter(OldSelectedNodes); SelectedIter; ++SelectedIter)
	{
		UEdGraphNode* Node = Cast<UEdGraphNode>(*SelectedIter);
		if (Node && Node->CanDuplicateNode())
		{
			CurrentGraphEditor->SetNodeSelection(Node, true);
		}
	}

	// Delete the duplicatable nodes
	DeleteSelectedNodes();

	CurrentGraphEditor->ClearSelectionSet();

	for (FGraphPanelSelectionSet::TConstIterator SelectedIter(OldSelectedNodes); SelectedIter; ++SelectedIter)
	{
		if (UEdGraphNode* Node = Cast<UEdGraphNode>(*SelectedIter))
		{
			CurrentGraphEditor->SetNodeSelection(Node, true);
		}
	}
}

void FAssetEditor_PSE::CutSelectedNodes()
{
	CopySelectedNodes();
	DeleteSelectedDuplicatableNodes();
}

bool FAssetEditor_PSE::CanCutNodes()
{
	return CanCopyNodes() && CanDeleteNodes();
}

void FAssetEditor_PSE::CopySelectedNodes()
{
	// Export the selected nodes and place the text on the clipboard
	FGraphPanelSelectionSet SelectedNodes = GetSelectedNodes();

	FString ExportedText;

	for (FGraphPanelSelectionSet::TIterator SelectedIter(SelectedNodes); SelectedIter; ++SelectedIter)
	{
		UEdGraphNode* Node = Cast<UEdGraphNode>(*SelectedIter);
		if (Node == nullptr)
		{
			SelectedIter.RemoveCurrent();
			continue;
		}

		if (UEdNode_PSEEdge* EdNode_Edge = Cast<UEdNode_PSEEdge>(*SelectedIter))
		{
			UEdNode_PSENode* StartNode = EdNode_Edge->GetStartNode();
			UEdNode_PSENode* EndNode = EdNode_Edge->GetEndNode();

			if (!SelectedNodes.Contains(StartNode) || !SelectedNodes.Contains(EndNode))
			{
				SelectedIter.RemoveCurrent();
				continue;
			}
		}

		Node->PrepareForCopying();
	}

	FEdGraphUtilities::ExportNodesToText(SelectedNodes, ExportedText);
	FPlatformApplicationMisc::ClipboardCopy(*ExportedText);
}

bool FAssetEditor_PSE::CanCopyNodes()
{
	// If any of the nodes can be duplicated then we should allow copying
	const FGraphPanelSelectionSet SelectedNodes = GetSelectedNodes();
	for (FGraphPanelSelectionSet::TConstIterator SelectedIter(SelectedNodes); SelectedIter; ++SelectedIter)
	{
		UEdGraphNode* Node = Cast<UEdGraphNode>(*SelectedIter);
		if (Node && Node->CanDuplicateNode())
		{
			return true;
		}
	}

	return false;
}

void FAssetEditor_PSE::PasteNodes()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (CurrentGraphEditor.IsValid())
	{
		PasteNodesHere(CurrentGraphEditor->GetPasteLocation());
	}
}

void FAssetEditor_PSE::PasteNodesHere(const FVector2D& Location)
{
	// Find the graph editor with focus
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (!CurrentGraphEditor.IsValid())
	{
		return;
	}
	// Select the newly pasted stuff
	UEdGraph* EdGraph = CurrentGraphEditor->GetCurrentGraph();

	{
		const FScopedTransaction Transaction(FGenericCommands::Get().Paste->GetDescription());
		EdGraph->Modify();

		// Clear the selection set (newly pasted stuff will be selected)
		CurrentGraphEditor->ClearSelectionSet();

		// Grab the text to paste from the clipboard.
		FString TextToImport;
		FPlatformApplicationMisc::ClipboardPaste(TextToImport);

		// Import the nodes
		TSet<UEdGraphNode*> PastedNodes;
		FEdGraphUtilities::ImportNodesFromText(EdGraph, TextToImport, PastedNodes);

		//Average position of nodes so we can move them while still maintaining relative distances to each other
		FVector2D AvgNodePosition(0.0f, 0.0f);

		for (TSet<UEdGraphNode*>::TIterator It(PastedNodes); It; ++It)
		{
			UEdGraphNode* Node = *It;
			AvgNodePosition.X += Node->NodePosX;
			AvgNodePosition.Y += Node->NodePosY;
		}

		float InvNumNodes = 1.0f / float(PastedNodes.Num());
		AvgNodePosition.X *= InvNumNodes;
		AvgNodePosition.Y *= InvNumNodes;

		for (TSet<UEdGraphNode*>::TIterator It(PastedNodes); It; ++It)
		{
			UEdGraphNode* Node = *It;
			CurrentGraphEditor->SetNodeSelection(Node, true);

			Node->NodePosX = (Node->NodePosX - AvgNodePosition.X) + Location.X;
			Node->NodePosY = (Node->NodePosY - AvgNodePosition.Y) + Location.Y;

			Node->SnapToGrid(16);

			// Give new node a different Guid from the old one
			Node->CreateNewGuid();
		}
	}

	// Update UI
	CurrentGraphEditor->NotifyGraphChanged();

	UObject* GraphOwner = EdGraph->GetOuter();
	if (GraphOwner)
	{
		GraphOwner->PostEditChange();
		GraphOwner->MarkPackageDirty();
	}
}

bool FAssetEditor_PSE::CanPasteNodes()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (!CurrentGraphEditor.IsValid())
	{
		return false;
	}

	FString ClipboardContent;
	FPlatformApplicationMisc::ClipboardPaste(ClipboardContent);

	return FEdGraphUtilities::CanImportNodesFromText(CurrentGraphEditor->GetCurrentGraph(), ClipboardContent);
}

void FAssetEditor_PSE::DuplicateNodes()
{
	CopySelectedNodes();
	PasteNodes();
}

bool FAssetEditor_PSE::CanDuplicateNodes()
{
	return CanCopyNodes();
}

void FAssetEditor_PSE::GraphSettings()
{
	PropertyWidget->SetObject(EditingGraph);
}

bool FAssetEditor_PSE::CanGraphSettings() const
{
	return true;
}

void FAssetEditor_PSE::OnRenameNode()
{
	TSharedPtr<SGraphEditor> CurrentGraphEditor = GetCurrGraphEditor();
	if (CurrentGraphEditor.IsValid())
	{
		const FGraphPanelSelectionSet SelectedNodes = GetSelectedNodes();
		for (FGraphPanelSelectionSet::TConstIterator NodeIt(SelectedNodes); NodeIt; ++NodeIt)
		{
			UEdGraphNode* SelectedNode = Cast<UEdGraphNode>(*NodeIt);
			if (SelectedNode != NULL && SelectedNode->bCanRenameNode)
			{
				CurrentGraphEditor->IsNodeTitleVisible(SelectedNode, true);
				break;
			}
		}
	}
}

bool FAssetEditor_PSE::CanRenameNodes() const
{
	UEdGraph_PSE* EdGraph = Cast<UEdGraph_PSE>(EditingGraph->EdGraph);
	check(EdGraph != nullptr);

	UPuzzleSequencer* Graph = EdGraph->GetGraph();
	check(Graph != nullptr)

	return Graph->bCanRenameNode && GetSelectedNodes().Num() == 1;
}

void FAssetEditor_PSE::OnSelectedNodesChanged(const TSet<UObject*>& NewSelection)
{
	TArray<UObject*> Selection;

	for (UObject* SelectionEntry : NewSelection)
	{
		Selection.Add(SelectionEntry);
	}

	if (Selection.Num() == 0)
	{
		PropertyWidget->SetObject(EditingGraph);
	}
	else
	{
		PropertyWidget->SetObjects(Selection);
	}
}

void FAssetEditor_PSE::OnNodeDoubleClicked(UEdGraphNode* Node)
{
}

void FAssetEditor_PSE::OnFinishedChangingProperties(const FPropertyChangedEvent& PropertyChangedEvent)
{
	if (EditingGraph == nullptr)
	{
		return;
	}

	EditingGraph->EdGraph->GetSchema()->ForceVisualizationCacheClear();
}

void FAssetEditor_PSE::OnPackageSavedWithContext(const FString& InName, UPackage* InPackage, FObjectPostSaveContext InContext)
{
	RebuildGraph();
}

#undef LOCTEXT_NAMESPACE
