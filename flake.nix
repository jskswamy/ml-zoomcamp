{
  description = "Machine Learning Zoomcamp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    git-hooks,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            # Secrets detection (runs on commit and push)
            gitleaks = {
              enable = true;
              entry = "${pkgs.gitleaks}/bin/gitleaks protect --staged --redact --verbose";
              language = "system";
              pass_filenames = false;
              always_run = true;
              stages = ["pre-commit" "pre-push"];
            };
          };
        };
        basePackages = with pkgs; [
          gitleaks
          uv
          process-compose
          pandoc
        ];
      in {
        devShells = {
          default = pkgs.mkShell {
            name = "default";
            inherit (pre-commit-check) shellHook;
            buildInputs =
              basePackages
              ++ (with pkgs; [
                clang
              ])
              ++ pre-commit-check.enabledPackages;
          };
          ml = pkgs.mkShell {
            name = "ml";
            inherit (pre-commit-check) shellHook;
            buildInputs =
              basePackages
              ++ (with pkgs; [
                uv
              ])
              ++ pre-commit-check.enabledPackages;
          };
        };
      }
    );
}
