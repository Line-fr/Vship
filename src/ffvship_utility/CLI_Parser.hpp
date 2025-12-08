#pragma once

#include <map>
#include <variant>
#include <iomanip> 
#include <vector>

namespace helper{

struct ArgParser {
    using TargetVariant = std::variant<bool*, int*, std::string*>;

    struct FlagGroup {
        bool set = false;
        std::vector<std::string> aliases; 
        TargetVariant target; 
        std::string help_text;
    };

    std::string binary_name = "BinaryName";
    std::vector<FlagGroup> flag_groups;
    std::map<std::string, int> alias_map; //returns index in flag_groups
    std::vector<int> positional_indexing;
    uint positional_counter = 0;

    ArgParser() {
        add_flag({"-h", "--help"}, &show_help_flag, "Display this help message");
        add_flag({"--config-file"}, &config_filepath, "Allows to input CLI Arguments specifying additional options (parsed after CLI)");
    }

    template<typename TargetType>
    void add_flag(const std::vector<std::string>& flag_names, TargetType* target_pointer,
                  std::string help_description = "", bool positional = false) {
        static_assert(std::is_same_v<TargetType, bool> || std::is_same_v<TargetType, int> ||
            std::is_same_v<TargetType, std::string>, "Unsupported flag type"
        );

        for (const std::string& name : flag_names) alias_map[name] = flag_groups.size(); //new index
        if (positional) positional_indexing.push_back(flag_groups.size());
        flag_groups.push_back({false, flag_names, target_pointer, help_description});
        //const auto& group = flag_groups.back();
    }

    // Takes in a vector of string from the cli to parse, returns a 0 if sucessfull
    int parse_cli_args(const std::vector<std::string>& args) {
        binary_name = args[0];
        size_t current_arg_index = 1;
        while (current_arg_index < args.size()) {
            if (!parse_flag(args, current_arg_index)) {
                std::cerr << "Failed to parse argument: " << args[current_arg_index] << "\n";
                return 1;
            }
            ++current_arg_index;
        }
        if (!config_filepath.empty()){
            return parse_config();
        }
        if (show_help_flag || args.size() == 1) { print_help(); return 2; }
        return 0;
    }

    int parse_config(){
        std::ifstream file(config_filepath);
        if (!file){
            std::cerr << "Error opening config file " << config_filepath << std::endl;
            return 1;
        }

        std::string newcli_args;
        if (!std::getline(file, newcli_args)){
            std::cerr << "Error retrieving first line of config file " << config_filepath << std::endl;
            return 1;
        }
        std::vector<std::string> arguments(1, binary_name);
        std::string temp;
        for (const char c: newcli_args){
            switch (c){
                case ' ':
                    if (!temp.empty()) arguments.push_back(temp);
                    temp.clear();
                    break;
                default:
                    temp.push_back(c);
            }
        }
        if (!temp.empty()) arguments.push_back(temp);
        config_filepath = ""; //we can't parse it twice.
        return parse_cli_args(arguments);
    }

private:
    mutable bool show_help_flag = false;
    std::string config_filepath;

    // Parse a flag at index in 'arguments', returns false if unknown or invalid
    bool parse_flag(const std::vector<std::string>& arguments, size_t& index) {
        const std::string& flag = arguments[index];

        FlagGroup* group_ptr; //output of the if
        bool positional = flag[0] != '-';
        if (!positional){
            auto foundIndexIterator = alias_map.find(flag);
            if (foundIndexIterator == alias_map.end()) {
                std::cerr << "Unknown argument: " << flag << "\n";
                return false;
            }
            group_ptr = &flag_groups[foundIndexIterator->second];
        } else { //positional
            if (positional_counter == positional_indexing.size()){
                std::cerr << "Too many Positional arguments provided at : " << flag << std::endl;
                return false;
            }
            group_ptr = &flag_groups[positional_indexing[positional_counter]];
            positional_counter++;
            index--; //a positional argument is itself, so for next part, we suppose the flag was behind and parse the flag as argument
        }
        if (group_ptr->set){
            std::cerr << "This Argument is already set : " << flag;
            if (group_ptr->aliases.size() > 0 && positional){ //no need to print that for non positional since flag is the alias
                std::cerr << " (Corresponding to " << group_ptr->aliases[0] << " )";
            }
            std::cerr << std::endl;
            return false;
        }
        group_ptr->set = true;
        const TargetVariant& found = group_ptr->target;
    
        if (std::holds_alternative<bool*>(found)) {
            bool* ptr = std::get<bool*>(found);
            *ptr = !*ptr;
            return true;
        }
    
        if (index + 1 >= arguments.size()) {
            std::cerr << flag << " requires an argument\n";
            return false;
        }
    
        if (std::holds_alternative<std::string*>(found)) {
            std::string* ptr = std::get<std::string*>(found);
            *ptr = arguments[++index];
            return true;
        }
    
        // Assume it is an int
        int* ptr = std::get<int*>(found);
        try {
            *ptr = std::stoi(arguments[++index]);
            return true;
        } catch (...) {
            std::cerr << "Invalid integer value: " << arguments[index] << " for arg "
                << flag << "\n";
            return false;
        }
    }

    void print_help() const {
        std::cout << "Usage : " << std::endl;
        std::cout << binary_name << " [";
        for (unsigned long pos_id = 0; pos_id < positional_indexing.size(); pos_id++){
            std::cout << "Positional " << pos_id;
            auto& group = flag_groups[positional_indexing[pos_id]];
            if (group.aliases.size() > 0){
                std::cout << " (Refering to " << group.aliases[0] << ")";
            }
            std::cout << "]";
            if (pos_id +1 < positional_indexing.size()){
                std::cout << " [";
            }
        }
        std::cout << " [Options]" << std::endl << std::endl;

        std::cout << "Available options:\n";

        size_t max_length = 0;
        std::vector<std::string> formatted_aliases;
        formatted_aliases.reserve(flag_groups.size());

        for (const auto& group : flag_groups) {
            std::string joined;
            for (size_t i = 0; i < group.aliases.size(); ++i) {
                joined += group.aliases[i];
                if (i + 1 < group.aliases.size()) joined += ", ";
            }
            max_length = std::max(max_length, joined.size());
            formatted_aliases.push_back(std::move(joined));
        }

        for (size_t i = 0; i < flag_groups.size(); ++i) {
            std::cout << "  " << std::left << std::setw(static_cast<int>(max_length))
                      << formatted_aliases[i] << "  " << flag_groups[i].help_text << "\n";
        }
    }
};

} //namespace