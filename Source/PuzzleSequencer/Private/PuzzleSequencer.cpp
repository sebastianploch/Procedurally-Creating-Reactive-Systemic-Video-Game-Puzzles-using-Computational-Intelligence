#include "PuzzleSequencer.h"
#include "Engine/Engine.h"

UPuzzleSequencer::UPuzzleSequencer()
{
	NodeType = UPuzzleSequencerNode::StaticClass();
	EdgeType = UPuzzleSequencerEdge::StaticClass();

	bEdgeEnabled = true;

#if WITH_EDITORONLY_DATA
	EdGraph = nullptr;
	bCanRenameNode = true;
#endif
}

void UPuzzleSequencer::Print(bool InToConsole, bool InToScreen)
{
	int level = 0;
	TArray<UPuzzleSequencerNode*> currentLevelNodes = RootNodes;
	TArray<UPuzzleSequencerNode*> nextLevelNodes{};

	while (currentLevelNodes.Num() != 0)
	{
		for (int i = 0; i < currentLevelNodes.Num(); ++i)
		{
			auto* node = currentLevelNodes[i];
			check(node != nullptr)

			const FString message = FString::Printf(TEXT("%s, Level %d"), *node->GetDescription().ToString(), level);
			if (InToScreen && GEngine)
			{
				GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::Blue, message);
			}

			for (int j = 0; j < node->ChildrenNodes.Num(); ++j)
			{
				nextLevelNodes.Add(node->ChildrenNodes[j]);
			}
		}

		currentLevelNodes = nextLevelNodes;
		nextLevelNodes.Reset();
		++level;
	}
}

int UPuzzleSequencer::GetLevelNum() const
{
	int level = 0;
	TArray<UPuzzleSequencerNode*> currentLevelNodes = RootNodes;
	TArray<UPuzzleSequencerNode*> nextLevelNodes{};

	while (currentLevelNodes.Num() != 0)
	{
		for (int i = 0; i < currentLevelNodes.Num(); ++i)
		{
			auto* node = currentLevelNodes[i];
			check(node != nullptr);

			for (int j = 0; j < node->ChildrenNodes.Num(); ++j)
			{
				nextLevelNodes.Add(node->ChildrenNodes[j]);
			}
		}

		currentLevelNodes = nextLevelNodes;
		nextLevelNodes.Reset();
		++level;
	}

	return level;
}

void UPuzzleSequencer::GetNodesByLevel(int InLevel, TArray<UPuzzleSequencerNode*>& OutNodes)
{
	int currentLevel = 0;
	TArray<UPuzzleSequencerNode*> nextLevelNodes{};

	OutNodes = RootNodes;

	while (OutNodes.Num() != 0)
	{
		if (currentLevel == InLevel)
		{
			break;
		}

		for (int i = 0; i < OutNodes.Num(); ++i)
		{
			auto* node = OutNodes[i];
			check(node != nullptr);

			for (int j = 0; j < node->ChildrenNodes.Num(); ++j)
			{
				nextLevelNodes.Add(node->ChildrenNodes[j]);
			}
		}

		OutNodes = nextLevelNodes;
		nextLevelNodes.Reset();
		++currentLevel;
	}
}

void UPuzzleSequencer::ClearGraph()
{
	for (int i = 0; i < AllNodes.Num(); ++i)
	{
		auto* node = AllNodes[i];
		if (node)
		{
			node->ParentNodes.Empty();
			node->ChildrenNodes.Empty();
			node->Edges.Empty();
		}
	}

	AllNodes.Empty();
	RootNodes.Empty();
}
