#pragma once

#include "CoreMinimal.h"
#include "Settings_PSE.h"
#include "PuzzleSequencer.h"

class PUZZLESEQUENCEREDITOR_API FAssetEditor_PSE : public FAssetEditorToolkit, public FNotifyHook, public FGCObject
{
public:
	FAssetEditor_PSE();
	virtual ~FAssetEditor_PSE() override;

	void InitPuzzleSequencerAssetEditor(const EToolkitMode::Type InMode, const TSharedPtr<IToolkitHost>& InInitToolkitHost, UPuzzleSequencer* InGraph);

	virtual void RegisterTabSpawners(const TSharedRef<FTabManager>& InTabManager) override;
	virtual void UnregisterTabSpawners(const TSharedRef<FTabManager>& InTabManager) override;

	virtual FName GetToolkitFName() const override;
	virtual FText GetBaseToolkitName() const override;
	virtual FText GetToolkitName() const override;
	virtual FText GetToolkitToolTipText() const override;
	virtual FLinearColor GetWorldCentricTabColorScale() const override;
	virtual FString GetWorldCentricTabPrefix() const override;
	virtual FString GetDocumentationLink() const override;
	virtual void SaveAsset_Execute() override;

	void UpdateToolbar();
	TSharedPtr<class FAssetEditorToolbar_PSE> GetToolbarBuilder() { return ToolbarBuilder; }
	void RegisterToolBarTab(const TSharedRef<class FTabManager>& InTabManager);

	virtual void AddReferencedObjects(FReferenceCollector& Collector) override;

	USettings_PSE* GetSettings() const;

protected:
	TSharedRef<SDockTab> SpawnTab_Viewport(const FSpawnTabArgs& Args);
	TSharedRef<SDockTab> SpawnTab_Details(const FSpawnTabArgs& Args);
	TSharedRef<SDockTab> SpawnTab_EditorSettings(const FSpawnTabArgs& Args);

	void CreateInternalWidgets();
	TSharedRef<SGraphEditor> CreateViewportWidget();


	void BindCommands();

	void CreateEdGraph();

	void CreateCommandList();

	TSharedPtr<SGraphEditor> GetCurrGraphEditor() const;

	FGraphPanelSelectionSet GetSelectedNodes() const;

	void RebuildGraph();

	// Delegates for graph editor commands
	void SelectAllNodes();
	bool CanSelectAllNodes();
	void DeleteSelectedNodes();
	bool CanDeleteNodes();
	void DeleteSelectedDuplicatableNodes();
	void CutSelectedNodes();
	bool CanCutNodes();
	void CopySelectedNodes();
	bool CanCopyNodes();
	void PasteNodes();
	void PasteNodesHere(const FVector2D& Location);
	bool CanPasteNodes();
	void DuplicateNodes();
	bool CanDuplicateNodes();

	void GraphSettings();
	bool CanGraphSettings() const;

	void OnRenameNode();
	bool CanRenameNodes() const;

	//////////////////////////////////////////////////////////////////////////
	// graph editor event
	void OnSelectedNodesChanged(const TSet<class UObject*>& NewSelection);

	void OnNodeDoubleClicked(UEdGraphNode* Node);

	void OnFinishedChangingProperties(const FPropertyChangedEvent& PropertyChangedEvent);

	void OnPackageSaved(const FString& PackageFileName, UObject* Outer);

protected:
	USettings_PSE* EditorSettings{nullptr};
	UPuzzleSequencer* EditingGraph{nullptr};

	// ToolBar
	TSharedPtr<class FAssetEditorToolbar_PSE> ToolbarBuilder{};

	FDelegateHandle OnPackageSavedDelegateHandle{};

	TSharedPtr<SGraphEditor> ViewportWidget{};
	TSharedPtr<class IDetailsView> PropertyWidget{};
	TSharedPtr<class IDetailsView> EditorSettingsWidget{};

	TSharedPtr<FUICommandList> GraphEditorCommands{};
};
