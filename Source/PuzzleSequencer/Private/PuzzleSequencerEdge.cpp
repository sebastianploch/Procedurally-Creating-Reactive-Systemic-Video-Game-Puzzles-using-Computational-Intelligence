#include "PuzzleSequencerEdge.h"

UPuzzleSequencer* UPuzzleSequencerEdge::GetGraph() const
{
	return Graph;
}

#if WITH_EDITOR
void UPuzzleSequencerEdge::SetNodeTitle(const FText& InNewTitle)
{
	NodeTitle = InNewTitle;
}
#endif
