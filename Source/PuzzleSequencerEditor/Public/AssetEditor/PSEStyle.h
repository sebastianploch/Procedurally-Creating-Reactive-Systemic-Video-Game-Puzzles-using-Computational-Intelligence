#pragma once

#include "CoreMinimal.h"
#include "Styling/SlateStyle.h"

class PUZZLESEQUENCEREDITOR_API FPSEStyle
{
public:
	static void Initialise();
	static void Shutdown();

	static const FName& GetStyleSetName();

private:
	inline static TSharedPtr<FSlateStyleSet> StyleSet{nullptr};
};
