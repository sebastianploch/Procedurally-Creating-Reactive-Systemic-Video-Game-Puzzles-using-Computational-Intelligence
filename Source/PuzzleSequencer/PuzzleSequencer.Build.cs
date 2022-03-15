using UnrealBuildTool;

public class PuzzleSequencer : ModuleRules
{
    public PuzzleSequencer(ReadOnlyTargetRules Target) : base(Target)
    {
        PrivateDependencyModuleNames.AddRange(new string[] {"Core"});
    }
}