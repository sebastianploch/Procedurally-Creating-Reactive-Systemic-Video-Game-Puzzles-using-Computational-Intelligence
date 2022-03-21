#pragma once

#include "CoreMinimal.h"

class PUZZLESEQUENCEREDITOR_API FEditorCommands_PSE : public TCommands<FEditorCommands_PSE>
{
public:
	FEditorCommands_PSE();

	TSharedPtr<FUICommandInfo> GraphSettings{};

	virtual void RegisterCommands() override;
};
