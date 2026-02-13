use clap::Parser;
use serde_json::Value;
use std::fs;

#[derive(Parser)]
#[command(name = "search")]
#[command(about = "Search for Pokémon cards by name and optionally by attack")]
struct Args {
    /// The search query to match against card names (case-insensitive)
    query: String,

    /// Optional attack name to filter Pokemon cards (case-insensitive)
    #[arg(short, long)]
    attack: Option<String>,

    /// Show only name and id instead of full card details
    #[arg(short, long)]
    simple: bool,

    /// Path to the database file
    #[arg(short, long, default_value = "database.json")]
    database: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read the database file
    let database_content = fs::read_to_string(&args.database)?;
    let database: Vec<Value> = serde_json::from_str(&database_content)?;

    // Search for matching cards
    let mut matches = Vec::new();
    let query_lower = args.query.to_lowercase();
    let attack_query_lower = args.attack.as_ref().map(|a| a.to_lowercase());

    for card in database {
        // Check if it's a Pokemon card
        if let Some(pokemon) = card.get("Pokemon") {
            if let Some(name) = pokemon.get("name").and_then(|n| n.as_str()) {
                if name.to_lowercase().contains(&query_lower) {
                    // If attack filter is specified, check if Pokemon has matching attack
                    let mut attack_match = true;
                    if let Some(ref attack_query) = attack_query_lower {
                        attack_match = false;
                        if let Some(attacks) = pokemon.get("attacks").and_then(|a| a.as_array()) {
                            for attack in attacks {
                                if let Some(title) = attack.get("title").and_then(|t| t.as_str()) {
                                    if title.to_lowercase().contains(attack_query) {
                                        attack_match = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if attack_match {
                        if args.simple {
                            // Create a simplified version with just id and name
                            let simplified = serde_json::json!({
                                "Pokemon": {
                                    "id": pokemon.get("id"),
                                    "name": name
                                }
                            });
                            matches.push(simplified);
                        } else {
                            matches.push(card);
                        }
                    }
                }
            }
        }
        // Check if it's a Trainer card (only if no attack filter is specified)
        else if let Some(trainer) = card.get("Trainer") {
            // Only include Trainer cards if no attack filter is specified
            if attack_query_lower.is_none() {
                if let Some(name) = trainer.get("name").and_then(|n| n.as_str()) {
                    if name.to_lowercase().contains(&query_lower) {
                        if args.simple {
                            // Create a simplified version with just id and name
                            let simplified = serde_json::json!({
                                "Trainer": {
                                    "id": trainer.get("id"),
                                    "name": name
                                }
                            });
                            matches.push(simplified);
                        } else {
                            matches.push(card);
                        }
                    }
                }
            }
        }
    }

    // Output the results as JSON
    let output = serde_json::to_string_pretty(&matches)?;
    println!("{output}");

    Ok(())
}
