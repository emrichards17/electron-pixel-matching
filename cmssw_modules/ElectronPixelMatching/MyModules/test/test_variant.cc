#include <any>
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

typedef std::vector<float> vfloat;
typedef std::vector<std::vector<float>> vvfloat;

template<class>
inline constexpr bool always_false_v = false;


template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << ", ";
    }
    os << "]";
    return os;
}

class TreeOutput
{
    public :
    
    std::unordered_map <std::string, std::variant<int, float, vfloat, vvfloat>> m_content;
    
    template<typename T>
    void registerVar(std::string varName)
    {
        m_content[varName] = T();
    }
    
    /*template<typename T1, typename T2>
    void fillVar(T1& var, const T2& fillContent)
    {
        var.push_back(fillContent);
    }
    
    template<typename T>
    void fillVar(const std::string& varName, const T& fillContent)
    {
        auto& content = m_content[varName];
        
        std::visit([&fillContent, this](auto&& arg)
        {
            using TC = std::decay_t<decltype(arg)>;
            
            if constexpr (std::is_same_v<TC, int>)
            {
                std::cout << "fillVar<int>(...): " << fillContent << "\n";
                arg = fillContent;
            }
            else if constexpr (std::is_same_v<TC, float>)
            {
                std::cout << "fillVar<float>(...): " << fillContent << "\n";
                arg = fillContent;
            }
            else if constexpr (std::is_same_v<TC, vfloat>)
            {
                //arg.push_back(fillContent);
                std::cout << "fillVar<vfloat, float>(...): " << fillContent << "\n";
                fillVar<vfloat, float>(arg, fillContent);
            }
            else if constexpr (std::is_same_v<TC, vvfloat>)
            {
                //arg.push_back(fillContent);
                std::cout << "fillVar<vvfloat, vfloat>(...): " << fillContent << "\n";
                fillVar<vvfloat, vfloat>(arg, fillContent);
            }
            else
            {
                static_assert(always_false_v<TC>, "non-exhaustive visitor!");
            }
        }, content);
    }*/
    
    template<typename T>
    void setVar(const std::string& name, const T& value)
    {
        assert(m_content.find(name) != m_content.end());
        m_content[name] = value;
    }
    
    template<typename T>
    void fillVarV(const std::string& name, const T& value)
    {
        assert(m_content.find(name) != m_content.end());
        auto& content = std::get<std::vector<T>>(m_content[name]);
        //(&content)->push_back(value);
        content.push_back(value);
    }
    
    template<typename T>
    void fillVarVV(const std::string& name, const std::vector<T>& value)
    {
        assert(m_content.find(name) != m_content.end());
        auto& content = std::get<std::vector<std::vector<T>>>(m_content[name]);
        content.push_back(value);
    }
    
    void clearVars()
    {
        for (auto& [ key, content ] : m_content)
        {
            std::visit([&key](auto&& arg)
            {
                using TC = std::decay_t<decltype(arg)>;
                
                if constexpr (std::is_same_v<TC, int>)
                {
                    std::cout << "clearVars(...): " << key << "\n";
                    arg = 0;
                }
                else if constexpr (std::is_same_v<TC, float>)
                {
                    std::cout << "clearVars(...): " << key << "\n";
                    arg = 0;
                }
                else if constexpr (std::is_same_v<TC, vfloat>)
                {
                    std::cout << "clearVars(...): " << key << "\n";
                    arg.clear();
                }
                else if constexpr (std::is_same_v<TC, vvfloat>)
                {
                    std::cout << "clearVars(...): " << key << "\n";
                    arg.clear();
                }
                else
                {
                    static_assert(always_false_v<TC>, "non-exhaustive visitor!");
                }
            }, content);
        }
    }
    
    TreeOutput()
    {
        registerVar<int>("runNumber");
        
        registerVar<vfloat>("ele_energy");
        
        registerVar<vvfloat>("ele_hit_x");
    }
};


int main() {
    std::printf("Start\n\n");
    
    TreeOutput *output = new TreeOutput();
    
    output->clearVars();
    
    for(int i = 0; i < 5; i++)
    {
        output->clearVars();
        
        output->setVar<int>("runNumber", i);
        
        for(int j = 0; j < 10; j++)
        {
            output->fillVarV<float>("ele_energy", (float)(i+(0.1*j)));
            
            vfloat v_tmp;
            
            for(int k = 0; k < 20; k++)
            {
                v_tmp.push_back(i+(0.1*j)+(0.01*k));
            }
            
            output->fillVarVV<float>("ele_hit_x", v_tmp);
        }
        
        std::cout << std::get<int>(output->m_content["runNumber"]) << "\n";
        std::cout << std::get<vfloat>(output->m_content["ele_energy"]) << "\n";
        std::cout << std::get<vvfloat>(output->m_content["ele_hit_x"]) << "\n";
    }
    
    std::printf("\nEnd\n");
    
    return 0;
}
