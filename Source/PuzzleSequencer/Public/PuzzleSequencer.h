#pragma once

#include "CoreMinimal.h"
#include "PuzzleSequencerNode.h"
#include "PuzzleSequencerEdge.h"
#include "GameplayTagContainer.h"
#include "PuzzleSequencer.generated.h"

UCLASS(Blueprintable)
class PUZZLESEQUENCER_API UPuzzleSequencer : public UObject
{
	GENERATED_BODY()

public:
	UPuzzleSequencer();

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencer")
	FString Name{};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencer")
	TSubclassOf<UPuzzleSequencerNode> NodeType{};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencer")
	TSubclassOf<UPuzzleSequencerEdge> EdgeType{};

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category="PuzzleSequencer")
	FGameplayTagContainer GraphTags{};

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencer")
	TArray<UPuzzleSequencerNode*> RootNodes{};

	UPROPERTY(BlueprintReadOnly, Category="PuzzleSequencer")
	TArray<UPuzzleSequencerNode*> AllNodes{};

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category="PuzzleSequencer")
	bool bEdgeEnabled{true};

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencer")
	void Print(bool InToConsole = true, bool InToScreen = true);

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencer")
	int GetLevelNum() const;

	UFUNCTION(BlueprintCallable, Category="PuzzleSequencer")
	void GetNodesByLevel(int InLevel, TArray<UPuzzleSequencerNode*>& OutNodes);

	void ClearGraph();

#pragma region Editor
#if WITH_EDITORONLY_DATA
	UPROPERTY()
	class UEdGraph* EdGraph{nullptr};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencer_Editor")
	bool bCanRenameNode{true};

	UPROPERTY(EditDefaultsOnly, Category="PuzzleSequencer_Editor")
	bool bCanBeCyclical{true};
#endif
#pragma endregion  Editor
};
